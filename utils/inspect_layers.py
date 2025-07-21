"""
# Usage
python inspect_layers.py
"""

import onnx

def inspect_onnx_model(model_path):
    """
    Loads an ONNX model and prints its input and output tensor details.

    Args:
        model_path (str): The path to the ONNX model file.
    """
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        print(f"Successfully loaded ONNX model: {model_path}")

        # Check model validity (optional, but good practice)
        onnx.checker.check_model(model)
        print("Model is valid (according to ONNX checker).")

        # Get graph inputs
        print("\n--- Model Inputs ---")
        if model.graph.input:
            for input_tensor in model.graph.input:
                name = input_tensor.name
                data_type = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
                shape = [dim.dim_value if dim.dim_value > 0 else "None" for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"  Name: {name}, Type: {data_type}, Shape: {shape}")
        else:
            print("  No explicit inputs found in the graph.")

        # Get graph outputs
        print("\n--- Model Outputs ---")
        if model.graph.output:
            for output_tensor in model.graph.output:
                name = output_tensor.name
                data_type = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
                shape = [dim.dim_value if dim.dim_value > 0 else "None" for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"  Name: {name}, Type: {data_type}, Shape: {shape}")
        else:
            print("  No explicit outputs found in the graph.")

        # You can also inspect nodes/layers
        print("\n--- First 5 Nodes/Layers (for a quick look) ---")
        for i, node in enumerate(model.graph.node):
            if i >= 5: # Limit to first 5 for brevity
                break
            print(f"  Node {i}: Name: {node.name if node.name else 'N/A'}, Op Type: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

    except Exception as e:
        print(f"Error loading or inspecting the ONNX model: {e}")


if __name__ == "__main__":
    # args = parse_args()
    # main(args)

    # Specify the path to your ONNX model
    # model_path = "/home/copter/onnx_models/FastSAM-s.onnx"
    model_path = "/home/copter/onnx_models/FastSAM-x.onnx"

    # Run the inspection
    inspect_onnx_model(model_path)