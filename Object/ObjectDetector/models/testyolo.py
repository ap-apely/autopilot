import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# Load the TensorRT engine
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    return engine

# Prepare input
def prepare_input(image):
    # Preprocess the image to match model input
    # Convert image to float32 and normalize if necessary
    input_data = image.astype(np.float32) / 255.0
    return input_data

# Run inference
def infer(engine, input_data):
    context = engine.create_execution_context()
    input_shape = (1, 3, 640, 640)  # Adjust according to your model's input shape
    input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize

    # Allocate buffers
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(engine.get_binding_shape(1).numel * np.dtype(np.float32).itemsize)

    # Transfer input data to the GPU
    cuda.memcpy_htod(d_input, input_data)
    
    # Execute inference
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

    # Retrieve output data from the GPU
    output_data = np.empty(engine.get_binding_shape(1).numel, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)

    return output_data

# Load the TensorRT engine
engine = load_engine('yolov10n.trt')

# Prepare your input image
image = np.random.rand(640, 640, 3)  # Replace with your actual image

# Prepare input data
input_data = prepare_input(image)

# Run inference
output = infer(engine, input_data)
print(output)
