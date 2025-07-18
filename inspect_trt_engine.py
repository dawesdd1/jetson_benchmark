"""
You need to inspect your mobile_sam_encoder_fp16.engine file to discover its actual input and output tensor names. 
We can do this using the tensorrt Python API.

Run the following Python script within your NanoSAM Conda environment.
This script will load your engine and print its input and output binding names.

EXPECTED:
>>> Inspecting TensorRT engine: /home/copter/onnx_models/mobile_sam_encoder_fp16.engine
>>> 
>>> Engine loaded successfully. Number of I/O Tensors: 2
>>>   Input: input | Shape: (1, 3, 1024, 1024) | Dtype: DataType.FLOAT
>>>   Output: output | Shape: (1, 256, 64, 64) | Dtype: DataType.FLOAT
>>> 
>>> Detected Input Names: ['input']
>>> Detected Output Names: ['output']

"""

import tensorrt as trt

def inspect_engine(engine_path):
    print(f"Inspecting TensorRT engine: {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Use WARNING to suppress verbose messages
    
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print(f"ERROR: Failed to deserialize engine from {engine_path}")
            return

        print(f"\nEngine loaded successfully. Number of I/O Tensors: {engine.num_io_tensors}")
        
        input_names = []
        output_names = []
        
        # Iterate through all I/O tensors
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            tensor_mode = engine.get_tensor_mode(tensor_name) # INPUT or OUTPUT
            
            if tensor_mode == trt.TensorIOMode.INPUT:
                input_names.append(tensor_name)
                print(f"  Input: {tensor_name} | Shape: {tensor_shape} | Dtype: {tensor_dtype}")
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)
                print(f"  Output: {tensor_name} | Shape: {tensor_shape} | Dtype: {tensor_dtype}")
        
        print(f"\nDetected Input Names: {input_names}")
        print(f"Detected Output Names: {output_names}")

if __name__ == "__main__":
    image_encoder_engine_path = "/home/copter/onnx_models/mobile_sam_encoder_fp16.engine"
    mask_decoder_engine_path = "/home/copter/onnx_models/mobile_sam_mask_decoder_fp16.engine"

    # Inspect the image encoder engine first, as that's where the current error is.
    # We might need to inspect the mask decoder later.
    inspect_engine(image_encoder_engine_path)
    # inspect_engine(mask_decoder_engine_path) # Uncomment this later if needed