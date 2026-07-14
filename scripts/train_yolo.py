# -*- coding: utf-8 -*-
"""
train_yolo.py
=============
대회 테스트 기간 중 촬영한 이미지로 YOLO 모델을 학습시키는 진입점입니다.
학습이 끝나면 나온 가중치(.pt) 경로를 config/field_config.py의
YOLO_OBJECT_WEIGHTS / YOLO_FACILITY_WEIGHTS에 넣고 DETECTOR_BACKEND(또는
FACILITY_BACKEND)를 "yolo"로 바꾸면 src/pipeline.py 수정 없이 바로 전환됩니다.

사용법:
  1) 폭파구/불발탄 통합 탐지 모델 (object detection, 클래스는
     field_config.YOLO_OBJECT_CLASS_MAP 순서와 반드시 일치시킬 것: big/medium/small + missile/dumb/cluster)
     python scripts/train_yolo.py --task detect --data data/object.yaml \
         --model yolo11n.pt --epochs 100 --out models/yolo11n_object

  2) 시설물 상태 분류 모델 (classification: normal/destroy/fire 폴더로 이미지 정리)
     python scripts/train_yolo.py --task classify --data data/facility_cls \
         --model yolo11n-cls.pt --epochs 50 --out models/yolo11n_facility_cls

ultralytics는 이 스크립트를 실제로 실행할 때만 필요합니다
(requirements.txt의 주석 처리된 줄을 해제하고 `pip install ultralytics`).

data.yaml/폴더 구성 방법은 ultralytics 공식 문서를 따르면 됩니다:
  - detect: images/, labels/ (YOLO txt 포맷) + data.yaml
  - classify: <root>/train/<클래스명>/*.jpg, <root>/val/<클래스명>/*.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detect", "classify"], required=True,
                         help="detect=폭파구/불발탄 통합 탐지, classify=시설물 상태 분류")
    parser.add_argument("--data", required=True,
                         help="detect: data.yaml 경로 / classify: train·val 폴더가 있는 루트 경로")
    parser.add_argument("--model", default=None,
                         help="시작 가중치 (기본: detect=yolo11n.pt, classify=yolo11n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=None,
                         help="학습된 가중치를 복사해둘 경로 (확장자 없이, .pt 자동 추가)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics가 설치되어 있지 않습니다. `pip install ultralytics` 후 다시 실행하세요.")
        sys.exit(1)

    default_model = "yolo11n.pt" if args.task == "detect" else "yolo11n-cls.pt"
    model = YOLO(args.model or default_model)

    print(f"[train_yolo] task={args.task} data={args.data} model={args.model or default_model} "
          f"epochs={args.epochs}")
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"학습 완료. 가중치: {best_weights}")

    if args.out:
        import shutil
        out_path = args.out if args.out.endswith(".pt") else args.out + ".pt"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.copyfile(best_weights, out_path)
        print(f"가중치를 {out_path} 로 복사했습니다.")
        cfg_key = "YOLO_OBJECT_WEIGHTS" if args.task == "detect" else "YOLO_FACILITY_WEIGHTS"
        print(f"config/field_config.py의 {cfg_key} = {out_path!r} 로 갱신하고, "
              f"{'DETECTOR_BACKEND' if args.task == 'detect' else 'FACILITY_BACKEND'} = \"yolo\" 로 바꾸세요.")


if __name__ == "__main__":
    main()
