import os
import torch
import torch_tensorrt
from mobile_sam import sam_model_registry
import torch.nn.functional as F
import torch.export

class TRTWrapper(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load the SAM model
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        
        # Store the sub-models that we will call directly
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.img_size = sam.image_encoder.img_size

    def forward(self, image, point_coords, point_labels):
        """
        A pure tensor-based forward pass that is traceable by torch_tensorrt.
        
        Args:
            image (torch.Tensor): [B, 3, H, W]
            point_coords (torch.Tensor): [B, N, 2]
            point_labels (torch.Tensor): [B, N]
        """
        # 1. Image Encoding
        # The image encoder (a ViT) processes the batched images directly.
        image_embeddings = self.image_encoder(image)

        # 2. Prompt Encoding
        # The prompt encoder is called with batched points and labels.
        # It returns sparse embeddings for the points.
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # 3. Mask Decoding
        # The mask decoder uses the image and prompt embeddings to predict masks.
        # All inputs are batched tensors.
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 4. Upscale Masks
        # The raw output masks are low-resolution. We upscale them to the
        # model's image input size using a standard tensor operation.
        masks = F.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # The outputs are now batched tensors, ready for use.
        return masks, iou_predictions

def compile_to_trt(ckpt_path, engine_path="mobile_sam.engine"):
    wrapper = TRTWrapper(ckpt_path).cuda().eval()

    # Sanity check on dummy data:
    B, H, W = 1, 1024, 1024
    # Use optimal shape for dummy data to ensure tracing works as expected.
    opt_num_points = 3
    dummy_img   = torch.randn(B, 3, H, W, device="cuda")
    dummy_coords= torch.randn(B, opt_num_points, 2, device="cuda")
    dummy_labels= torch.ones(B, opt_num_points, dtype=torch.int32, device="cuda")
    
    print("Running sanity check with dummy data...")
    try:
        with torch.no_grad():
            wrapper(dummy_img, dummy_coords, dummy_labels)
        print("✅ Sanity check passed.")
    except Exception as e:
        print(f"❌ Sanity check failed: {e}")
        return

    # 1. Define a symbolic dimension for the number of points (N).
    # This explicitly tells the tracer that this dimension is dynamic and, crucially,
    # that it is the *same* dimension for both point_coords and point_labels.
    num_points = torch.export.Dim("num_points", min=1, max=10)

    # 2. Define dynamic shapes using the symbolic dimension for torch.export.
    # The keys correspond to the arguments of the `forward` method.
    # The values map the axis index to the symbolic dimension.
    # The image input is static, so it's marked as None.
    dynamic_shapes = {
        "image": None,
        "point_coords": {1: num_points},  # Batch is dim 0, num_points is dim 1
        "point_labels": {1: num_points},  # Batch is dim 0, num_points is dim 1
    }
    
    # 3. Export the model with the dynamic shape constraints.
    # This creates a graph where the constraint is formally captured, satisfying Dynamo.
    print("Exporting the model with dynamic shape constraints...")
    exported_program = torch.export.export(
        wrapper,
        args=(dummy_img, dummy_coords, dummy_labels),
        dynamic_shapes=dynamic_shapes
    )

    # 4. Define the input specifications for the TensorRT profile.
    # This is still needed to tell TensorRT about the min, opt, and max bounds
    # of the dynamic dimension we defined.
    inputs = [
        torch_tensorrt.Input(
            name="image",
            dtype=torch.float32,
            min_shape=(1, 3, H, W),
            opt_shape=(1, 3, H, W),
            max_shape=(1, 3, H, W),
        ),
        torch_tensorrt.Input(
            name="point_coords",
            dtype=torch.float32,
            min_shape=(1, 1, 2),
            opt_shape=(1, opt_num_points, 2),
            max_shape=(1, 10, 2),
        ),
        torch_tensorrt.Input(
            name="point_labels",
            dtype=torch.int32,
            min_shape=(1, 1),
            opt_shape=(1, opt_num_points),
            max_shape=(1, 10),
        ),
    ]

    # 5. Compile the exported program with TensorRT.
    print("Compiling the exported program to TensorRT...")
    trt_mod = torch_tensorrt.compile(
        exported_program, # Pass the constrained, exported program
        inputs=inputs,    # Provide the profile shapes for TensorRT optimization
        enabled_precisions={torch.float32},
        workspace_size=1<<30,
    )
    
    # Serialize and save the engine
    with open(engine_path, "wb") as f:
        f.write(trt_mod.engine.serialize())
    print(f"✅ Wrote engine to {engine_path}")


if __name__ == "__main__":
    # IMPORTANT: Make sure this path is correct for your system
    ckpt = os.path.expanduser(
        "~/repos/MobileSAM/weights/mobile_sam.pt"
    )
    if not os.path.exists(ckpt):
        print(f"❌ Checkpoint file not found at: {ckpt}")
    else:
        compile_to_trt(ckpt)
