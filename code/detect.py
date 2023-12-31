"""
A demo which runs object detection on camera frames using GStreamer.
It also provides support for Object Tracker.

Run default object detection:
python3 detect.py

Choose different camera and input encoding
python3 detect.py --videosrc /dev/video1 --videofmt jpeg

Choose an Object Tracker. Example : To run sort tracker
python3 detect.py --tracker sort

TEST_DATA=../all_models

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import collections
import common
import gstreamer
import numpy as np
import os
import re
import svgwrite
import time

import kuksa
import kuksa_viss_client
import seeed_python_reterminal.core as rt
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

to_detect = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]
to_recognize = {"speed_limit_100":"100","speed_limit_120":"120","speed_limit_20":"20","speed_limit_30":"30", "speed_limit_40":"40", "speed_limit_50":"50", "speed_limit_60":"60", "speed_limit_70":"70", "speed_limit_80":"80", "traffic_sign_90":"90"}
PATH_OBSTACLE = "Vehicle.ADAS.ObstacleDetection.IsWarning"
PATH_SPEEDDETECTION = "Vehicle.ADAS.SpeedSign"

def obstacle_detected():
    client = kuksa.kuksa_ini()
    client.setValue(PATH_OBSTACLE, "true")
    client.stop()

def no_obstacle_detected():
    client = kuksa.kuksa_ini()
    client.setValue(PATH_OBSTACLE, "false")
    client.stop()

def is_close_by(labels, obj, min_close_by= 0.1, min_height_pct=0.6):
    x0, y0, x1, y1 = list(obj.bbox)
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    return labels.get(obj.id, obj.id) in to_detect and h > min_height_pct and y1 > min_close_by

def speed_recognize(labels, objs):
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        if label in to_recognize.keys():
            client = kuksa.kuksa_ini()
            client.setValue(PATH_SPEEDDETECTION, to_recognize[label])
            client.stop()
            break
    
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))


def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines):
    dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y*20, line)
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        # Relative coordinates.
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        # Absolute coordinates, input tensor space.
        x, y, w, h = int(x * inf_w), int(y *
                                            inf_h), int(w * inf_w), int(h * inf_h)
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y
        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        shadow_text(dwg, x, y - 5, label)
        dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                            fill='none', stroke='red', stroke_width='2'))
    return dwg.tostring()


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()
    
    def scale(self, sx, sy):
        """Scales the bounding box.

        Args:
        sx (float): Scale factor for the x-axis.
        sy (float): Scale factor for the y-axis.

        Returns:
        A :obj:`BBox` object with the rescaled dimensions.
        """
        return BBox(
            xmin=sx * self.xmin,
            ymin=sy * self.ymin,
            xmax=sx * self.xmax,
            ymax=sy * self.ymax)
    def map(self, f):
        """Maps all box coordinates to a new position using a given function.

        Args:
        f: A function that takes a single coordinate and returns a new one.

        Returns:
        A :obj:`BBox` with the new coordinates.
        """
        return BBox(
            xmin=f(self.xmin),
            ymin=f(self.ymin),
            xmax=f(self.xmax),
            ymax=f(self.ymax))

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter,1)
    category_ids = common.output_tensor(interpreter, 3)
    scores = common.output_tensor(interpreter, 0)
    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def main():
    default_model_dir = '../models'
    default_model = 'mobilenet_ssd_coco/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
    default_labels = 'mobilenet_ssd_coco/coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--do_sink', action="store_true", help='Flag to streanming on X11.',
                        default=False,)
    parser.add_argument('--do_detect', action="store_true", help='Flag to detect obstacles.',
                        default=True,)
    parser.add_argument('--do_recognize', action="store_true", help='Flag to recognize signs.',
                        default=False,)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    sink = args.do_sink
    do_detect = False if args.do_recognize else args.do_detect
    do_recognize = args.do_recognize
    model_path = args.model
    labels_path = args.labels
    if do_recognize:
        model_path = os.path.join(default_model_dir, 'eficientnet/efficientnet0_quantized_edgetpu.tflite')
        labels_path = os.path.join(default_model_dir, 'road_signs_labels.txt')
    interpreter = common.make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = load_labels(labels_path)

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    fps_counter = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box):
        nonlocal fps_counter
        nonlocal sink
        nonlocal labels
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_output(interpreter, args.threshold, args.top_k)
        end_time = time.monotonic()
        detections = []  # np.array([])
        for n in range(0, len(objs)):
            element = []  # np.array([])
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            element.append(objs[n].score)  # print('element= ',element)
            detections.append(element)  # print('dets: ',dets)
        # convert to numpy array #      print('npdets: ',dets)
        detections = np.array(detections)
        if detections.any():
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(fps_counter))), ]
        if len(objs) != 0:
            if not sink:
                if do_detect:
                    if any(is_close_by(labels, obj) for obj in objs):
                        print("obstacle detected !!")
                        obstacle_detected()
                        rt.buzzer = True
                    else:
                        no_obstacle_detected()
                        rt.buzzer = False
                elif do_recognize:
                    speed_recognize(labels ,objs)
                for line in text_lines:
                    print(line)
                for obj in objs:
                    print('Item: ', labels.get(obj.id, obj.id))
            else:
                return generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines)

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    sink=sink,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)


if __name__ == '__main__':
    main()
