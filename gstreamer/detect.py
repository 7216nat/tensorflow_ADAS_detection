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


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


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


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter,1)
    category_ids = common.output_tensor(interpreter, 3)
    scores = common.output_tensor(interpreter, 0)
    print(boxes)
    print(category_ids)
    print(scores)
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
    default_model = 'road_signs_quantized_edgetpu.tflite'
    default_labels = 'road_signs_labels_reduced.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--do_sink', action="store_true", help='Flag to streanming on X11.',
                        default=False,)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    sink = args.do_sink
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

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
        print("len: ", len(objs))
        if len(objs) != 0:
            if not sink:
                # TODO kuksa actions
                for line in text_lines:
                    print(line)
                for obj in objs:
                    print('Item: ', labels.get(obj.id, obj.id), )
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
