"""Common utilities."""
import collections
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import svgwrite
import tflite_runtime.interpreter as tflite
import time

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    if model_file.contains('edgetpu.tflite'):
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                     {'device': device[0]} if device else {})
            ])
    else:
        return tflite.Interpreter(model_path=model_file)

def input_image_size(interpreter):
    """Returns input size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, channels)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, buf):
    """Copies data to input tensor."""
    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if result:
        np_buffer = np.reshape(np.frombuffer(mapinfo.data, dtype=np.uint8),
            interpreter.get_input_details()[0]['shape'])
        input_tensor(interpreter)[:, :] = np_buffer
        buf.unmap(mapinfo)

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)
