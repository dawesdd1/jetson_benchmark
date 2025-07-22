"python /home/copter/jetson_benchmark/utils/engine_fp_inspect.py"

import tensorrt as trt

# engine_path = "your_model.engine"
engine_path = "/home/copter/engine_models/nvidia_nanosam_resnet18_image_encoder_fp16.engine"
logger = trt.Logger(trt.Logger.VERBOSE) # Use VERBOSE for more details

with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

    # Create an inspector
    inspector = engine.create_engine_inspector()

    # Iterate through layers and print their precision
    print("Engine Layer Information:")
    for i in range(engine.num_layers):
        layer = inspector.get_layer_information(i)
        print(f"  Layer {i}: Name={layer.name}, Type={layer.type}, Precision={layer.precision}")

    # You can also inspect input/output tensor data types
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        kind = engine.get_binding_is_input(i)
        print(f"  Binding {i}: Name={name}, Type={'Input' if kind else 'Output'}, DType={dtype}")