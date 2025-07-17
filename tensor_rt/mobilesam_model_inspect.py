import torch
from mobile_sam import sam_model_registry

model = sam_model_registry["vit_t"](checkpoint="/home/dawesdd1/repos/MobileSAM/weights/mobile_sam.pt")
print("Model components:")
print(f"- Image encoder: {hasattr(model, 'image_encoder')}")
print(f"- Prompt encoder: {hasattr(model, 'prompt_encoder')}")
print(f"- Mask decoder: {hasattr(model, 'mask_decoder')}")

import inspect
print(f"Forward signature: {inspect.signature(model.forward)}")