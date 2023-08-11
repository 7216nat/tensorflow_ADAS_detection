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
    if 'edgetpu.tflite' in model_file:
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

def input_details(interpreter, key):
  """Gets a model's input details by specified key.

  Args:
    interpreter: The ``tf.lite.Interpreter`` holding the model.
    key (int): The index position of an input tensor.
  Returns:
    The input details.
  """
  return interpreter.get_input_details()[0][key]

def input_size(interpreter):
  """Gets a model's input size as (width, height) tuple.

  Args:
    interpreter: The ``tf.lite.Interpreter`` holding the model.
  Returns:
    The input tensor size as (width, height) tuple.
  """
  _, height, width, _ = input_details(interpreter, 'shape')
  return width, height


def set_resized_input(interpreter, size, resize):
    """Copies a resized and properly zero-padded image to a model's input tensor.

    Args:
        interpreter: The ``tf.lite.Interpreter`` to update.
        size (tuple): The original image size as (width, height) tuple.
        resize: A function that takes a (width, height) tuple, and returns an
        image resized to those dimensions.

    Returns:
        The resized tensor with zero-padding as tuple
        (resized_tensor, resize_ratio).
    """
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = np.reshape(result, (h, w, channel))
    return result, (scale, scale)
