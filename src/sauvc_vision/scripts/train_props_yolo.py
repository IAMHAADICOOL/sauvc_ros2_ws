#!/usr/bin/env python3
"""train_props_yolo.py — train the SAUVC prop detector (plain script, no ROS).

  pip install ultralytics
  # 1. annotate saved frames (labelImg / Roboflow / CVAT) in YOLO box format:
  #    dataset/images/{train,val}/*.jpg ; dataset/labels/{train,val}/*.txt
  # 2. python3 train_props_yolo.py --data ~/sauvc_dataset/dataset --epochs 120

Classes (bounding boxes; keep this order in every label file):
  0 gate            whole gate span (both posts + top bar; annotate even if a post
                    is cut off — partial gates are the servoing case that matters)
  1 orange_flare    2 flare_red    3 flare_yellow    4 flare_blue
  5 drum_red        6 drum_blue    7 golf_ball       (tiny — teaches close approach)

Upgrade path: once boxes work, a pose/keypoint model on the gate's 4 corners gives
full PnP pose; boxes + known sizes (as in gate_detector_node) are enough to start.
"""
import argparse, os, textwrap

CLASSES = ['gate', 'orange_flare', 'flare_red', 'flare_yellow', 'flare_blue',
           'drum_red', 'drum_blue', 'golf_ball']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dataset root (images/ labels/)')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--model', default='yolov8n.pt', help='n=nano fits the Jetson')
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.data))
    yaml_path = os.path.join(root, 'sauvc_props.yaml')
    with open(yaml_path, 'w') as f:
        f.write(textwrap.dedent(f"""\
            path: {root}
            train: images/train
            val: images/val
            names: {dict(enumerate(CLASSES))}
            """))
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, degrees=5, translate=0.1, scale=0.3,
                fliplr=0.5, hsv_h=0.03, hsv_s=0.5, hsv_v=0.5,   # lighting robustness
                project='sauvc_props', name='yolov8n')
    model.export(format='engine', half=True)   # TensorRT for the Jetson; falls back
                                               # to .onnx if TRT is unavailable
    print('done: weights in sauvc_props/yolov8n/weights/')

if __name__ == '__main__':
    main()
