# -*- coding: utf-8 -*-
"""
debug_visualize.py
===================
프레임 1장을 실제 파이프라인(2단계 워핑 -> 3-A zone crop -> YOLO 배치 추론)과 동일한 경로로
처리한 뒤, zone 그리드 + 탐지 결과를 그려 debug_viz/에 저장합니다. 탐지 결과가 이상해 보일 때
(중복 탐지, 오탐 등) 원인을 코드 안 보고도 눈으로 바로 확인하기 위한 용도입니다.

사용법:
  python scripts/debug_visualize.py --image test_images/drone_20260714142246_frame_0000.png \
      --weights models/yolov8n_object_v2.pt
"""
import os
import sys
import argparse

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import field_config as fc
from calibration import FieldCalibrator
from tiling import crop_zone_tiles, crop_facility_rois
from detection import build_object_detector, detect_zone_tiles, build_facility_classifier
from img_io import imread_safe, imwrite_safe

GRID_COLOR = (110, 96, 80)
DET_COLOR = (32, 107, 255)
FACILITY_STATUS_COLOR = {
    "normal": (60, 180, 60),
    "destroy": (0, 140, 255),
    "fire": (0, 0, 255),
    "unconfirmed": (150, 150, 150),
}


def zone_pixel_rect(zone_name, px_per_cm):
    b = fc.SEGMENTS[zone_name]
    x0 = int(round(b["x_min"] * px_per_cm))
    y0 = int(round(b["y_min"] * px_per_cm))
    x1 = int(round(b["x_max"] * px_per_cm))
    y1 = int(round(b["y_max"] * px_per_cm))
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="시각화할 프레임(워핑 전 원본) 이미지 경로")
    parser.add_argument("--weights", type=str, default=None,
                         help="yolo 백엔드 가중치 경로 (생략 시 field_config.YOLO_OBJECT_WEIGHTS)")
    parser.add_argument("--detector-backend", choices=["classical", "yolo"], default="yolo")
    parser.add_argument("--conf", type=float, default=None,
                         help="탐지 conf threshold (생략 시 field_config 기본값)")
    parser.add_argument("--facility-weights", type=str, default=None,
                         help="시설물 분류 yolo 백엔드 가중치 경로 (생략 시 field_config.YOLO_FACILITY_WEIGHTS)")
    parser.add_argument("--facility-backend", choices=["classical", "yolo"], default="yolo")
    parser.add_argument("--facility-conf", type=float, default=None,
                         help="시설물 분류 conf threshold (생략 시 field_config 기본값)")
    parser.add_argument("--no-facility", action="store_true", help="시설물 상태 분류는 건너뛰고 폭파구/불발탄만 시각화")
    parser.add_argument("--output-dir", default="debug_viz")
    args = parser.parse_args()

    frame = imread_safe(args.image)
    if frame is None:
        print(f"이미지를 읽을 수 없습니다: {args.image}")
        sys.exit(1)

    calibrator = FieldCalibrator()
    calibrator.calibrate_from_image(frame)
    topview, px_per_cm = calibrator.warp_to_topview(frame)

    detector_kwargs = {}
    if args.weights:
        detector_kwargs["weights_path"] = args.weights
    if args.conf is not None:
        detector_kwargs["conf_threshold"] = args.conf
    detector = build_object_detector(args.detector_backend, **detector_kwargs)

    tiles = crop_zone_tiles(topview, px_per_cm)
    detections_by_zone = detect_zone_tiles(detector, tiles)

    facility_status = {}
    if not args.no_facility:
        facility_kwargs = {}
        if args.facility_weights:
            facility_kwargs["weights_path"] = args.facility_weights
        if args.facility_conf is not None:
            facility_kwargs["conf_threshold"] = args.facility_conf
        facility_classifier = build_facility_classifier(args.facility_backend, **facility_kwargs)
        facility_rois = crop_facility_rois(topview, px_per_cm)
        facility_status = facility_classifier.classify_frame_batch(facility_rois)

    canvas = topview.copy()
    for zone_name, bounds in fc.SEGMENTS.items():
        x0 = int(round(bounds["x_min"] * px_per_cm))
        y0 = int(round(bounds["y_min"] * px_per_cm))
        x1 = int(round(bounds["x_max"] * px_per_cm))
        y1 = int(round(bounds["y_max"] * px_per_cm))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), GRID_COLOR, 1)
        cv2.putText(canvas, zone_name, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRID_COLOR, 1, cv2.LINE_AA)

    total = 0
    print(f"[debug_visualize] {args.image} -> px_per_cm={px_per_cm}")
    for zone_name, dets in detections_by_zone.items():
        if not dets:
            continue
        x0, y0, _, _ = zone_pixel_rect(zone_name, px_per_cm)
        for det in dets:
            cx, cy = det.center_px
            gx, gy = x0 + cx, y0 + cy
            r = max(det.equiv_diameter_px, 6) / 2
            cv2.circle(canvas, (int(gx), int(gy)), int(r), DET_COLOR, 2, cv2.LINE_AA)
            conf_str = f"{det.confidence:.2f}" if det.confidence is not None else "?"
            label = f"{det.subtype or '?'} {conf_str}"
            cv2.putText(canvas, label, (int(gx - r), int(gy - r - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DET_COLOR, 1, cv2.LINE_AA)
            total += 1
            print(f"  {zone_name}: {det.category}/{det.subtype} conf={conf_str}")

    if facility_status:
        print("[시설물 상태]")
        for slot, (status, conf) in facility_status.items():
            x0, y0, x1, y1 = zone_pixel_rect(slot, px_per_cm)
            color = FACILITY_STATUS_COLOR.get(status, FACILITY_STATUS_COLOR["unconfirmed"])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
            label = f"{slot}: {status} {conf:.2f}"
            cv2.putText(canvas, label, (x0 + 4, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            print(f"  {slot}: {status} conf={conf:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.output_dir, f"{base}_debug.png")
    imwrite_safe(out_path, canvas)
    print(f"\n총 탐지 {total}건. 저장: {out_path}")


if __name__ == "__main__":
    main()
