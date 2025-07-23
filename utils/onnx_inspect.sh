python -c "
import onnx
model = onnx.load('/home/copter/onnx_models/FastSAM-x.onnx')
print('=== INPUT SHAPES ===')
for input in model.graph.input:
    print(f'Input: {input.name}')
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input.type.tensor_type.shape.dim]
    print(f'  Shape: {shape}')
    print(f'  Data type: {input.type.tensor_type.elem_type}')
    print()

print('=== OUTPUT SHAPES ===')
for output in model.graph.output:
    print(f'Output: {output.name}')
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output.type.tensor_type.shape.dim]
    print(f'  Shape: {shape}')
    print(f'  Data type: {output.type.tensor_type.elem_type}')
    print()
"