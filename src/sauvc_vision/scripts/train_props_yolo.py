#!/usr/bin/env python3
"""train_props_yolo.py — train the SAUVC prop detector (plain script, no ROS).

  pip install ultralytics
  # 1. annotate saved frames (labelImg / Roboflow / CVAT) in YOLO box format:
  #    dataset/images/{train,val}/*.jpg ; dataset/labels/{train,val}/*.txt
  # 2. python3 train_props_yolo.py --data ~/sauvc_dataset/dataset --epochs 120 \
  #        --target jetson   # or: --target laptop | --target none

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
    ap.add_argument('--target', choices=['jetson', 'laptop', 'none'], default='jetson',
                     help=("where the exported weights will run: "
                           "'jetson' -> TensorRT .engine, fp16 (needs TensorRT installed, "
                           "usually only present on the Jetson itself or via matching "
                           "TensorRT/CUDA versions on the training machine); "
                           "'laptop' -> .onnx, fp32 (portable, runs anywhere via onnxruntime, "
                           "no TensorRT required); 'none' -> skip export, keep the .pt weights"))
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.data))
    yaml_path = os.path.join(root, 'sauvc_props.yaml')
    with open(yaml_path, 'w') as f:
        f.write(textwrap.dedent(f"""\
            path: {root}
            train: train/images
            val: valid/images
            names: {dict(enumerate(CLASSES))}
            """))
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, degrees=5, translate=0.1, scale=0.3,
                patience=30,
                fliplr=0.5, hsv_h=0.03, hsv_s=0.5, hsv_v=0.5,   # lighting robustness
                project='sauvc_props', name='yolov8n')
    weights_dir = 'sauvc_props/yolov8n/weights/'
    if args.target == 'jetson':
        try:
            model.export(format='engine', half=True)   # TensorRT, fp16, fastest on Jetson
        except Exception as e:
            print(f'TensorRT export failed ({e}); falling back to ONNX. '
                  f'Build the .engine on the Jetson itself with:\n'
                  f'  yolo export model={weights_dir}best.pt format=engine half=True')
            model.export(format='onnx')
    elif args.target == 'laptop':
        model.export(format='onnx')   # portable, no TensorRT needed
    else:
        print('--target none: skipping export, .pt weights only')
    print(f'done: weights in {weights_dir}')

if __name__ == '__main__':
    main()
