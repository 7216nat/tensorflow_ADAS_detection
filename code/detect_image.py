"""

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

from PIL import Image
from PIL import ImageDraw

import detect


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}

def get_outputs(interpreter, score_threshold, top_k, image_scale=(1.0, 1.0)):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter,1)
    category_ids = common.output_tensor(interpreter, 3)
    scores = common.output_tensor(interpreter, 0)
    print(boxes)
    print(category_ids)
    print(scores)
    width, height = common.input_size(interpreter)
    image_scale_x, image_scale_y = image_scale
    sx, sy = width / image_scale_x, height / image_scale_y
    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=detect.BBox(xmin=xmin, ymin=ymin, xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))
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
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                      help='Score threshold for detected objects')
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    parser.add_argument('--input', required=True, help='Which image source to use. ')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    image = Image.open(args.input)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
            'loading the model into Edge TPU memory.')
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = get_outputs(interpreter, args.threshold, args.top_k, scale)
        print('%.2f ms' % (inference_time * 1000))
    
    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save(args.output)
        image.show()


if __name__ == '__main__':
    main()
