# -*- coding: utf-8 -*-
"""
run_mission.py
===============
전체 임무 파이프라인 실행 진입점.

사용법:
  1) 합성 테스트 이미지로 실행 (실제 드론 사진 없이 파이프라인 검증):
     python scripts/run_mission.py --synthetic

  2) 실제 촬영된 이미지 폴더로 실행:
     python scripts/run_mission.py --images /path/to/frames_dir

  3) YOLO 학습이 끝난 뒤 해당 실행만 YOLO 백엔드로 돌려보기
     (기본값은 field_config.DETECTOR_BACKEND/FACILITY_BACKEND):
     python scripts/run_mission.py --images /path/to/frames_dir --detector-backend yolo
"""
import os
import sys
import argparse
import json
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from pipeline import MissionPipeline
import field_config as fc


def load_frames_from_dir(images_dir):
    exts = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(exts)])
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(images_dir, f))
        if img is not None:
            frames.append(img)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="합성 테스트 이미지 생성 후 실행")
    parser.add_argument("--images", type=str, default=None, help="실제 이미지가 있는 폴더 경로")
    parser.add_argument("--output", type=str, default="output", help="결과 JSON 저장 폴더")
    parser.add_argument("--no-llm", action="store_true", help="로컬 LLM 시도 없이 템플릿 보고서만 사용")
    parser.add_argument("--send", action="store_true", help="파이프라인 완료 후 대시보드 전송(6단계)까지 실행")
    parser.add_argument("--detector-backend", choices=["classical", "yolo"], default=None,
                         help="폭파구/불발탄 탐지 백엔드 (생략 시 field_config.DETECTOR_BACKEND 사용)")
    parser.add_argument("--facility-backend", choices=["classical", "yolo"], default=None,
                         help="시설물 상태 분류 백엔드 (생략 시 field_config.FACILITY_BACKEND 사용)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.synthetic:
        sys.path.insert(0, os.path.join(base_dir, "scripts"))
        import generate_test_scene
        test_img_dir = os.path.join(base_dir, "test_images")
        print(f"[1/3] 합성 테스트 이미지 생성 중... -> {test_img_dir}")
        generate_test_scene.main(test_img_dir)
        frames = load_frames_from_dir(test_img_dir)
    elif args.images:
        frames = load_frames_from_dir(args.images)
    else:
        print("`--synthetic` 또는 `--images <폴더경로>` 중 하나를 지정하세요.")
        sys.exit(1)

    if not frames:
        print("불러올 이미지가 없습니다.")
        sys.exit(1)

    print(f"[2/3] 프레임 {len(frames)}장 로드 완료. 파이프라인 실행 중...")

    output_dir = os.path.join(base_dir, args.output)
    pipeline = MissionPipeline(mission_code=fc.MISSION_CODE, use_llm=not args.no_llm,
                                output_dir=output_dir,
                                detector_backend=args.detector_backend,
                                facility_backend=args.facility_backend)
    result = pipeline.run(frames, send_to_dashboard=args.send)

    print(f"[3/3] 완료! 결과 저장 위치: {output_dir}\n")
    print("=" * 60)
    print("검증 결과:", "통과" if result["validation"]["ok"] else "실패")
    if result["validation"]["errors"]:
        print("  오류:", result["validation"]["errors"])
    if result["validation"]["warnings"]:
        print("  경고:", result["validation"]["warnings"])
    print("=" * 60)
    print("소요시간(초):", json.dumps(result["timing"], ensure_ascii=False, indent=2))
    print("=" * 60)

    print("\n[핵심 결과 요약]")
    outputs = result["outputs"]
    print(f"  - 폭파구 총 개수: {outputs['crater_detect']['crater_count_total']}")
    print(f"  - 활주로 폭파구 개수: {outputs['crater_count']['runway_crater_count']}")
    print(f"  - 활주로 가용길이: {outputs['runway_status']['runway_available_length_m']} m")
    print(f"  - 최장 가용 구간: {outputs['runway_status']['longest_available_run']}")
    print(f"  - 막힌 구간: {outputs['runway_status']['blocked_segments']}")
    print(f"  - 불발탄 총 개수: {outputs['uxo_detect']['uxo_count_total']}")
    print(f"  - 활주로 불발탄 개수: {outputs['uxo_count']['runway_uxo_count']}")
    print(f"  - 시설물 상태:")
    for f in outputs["facility_status"]["facilities"]:
        print(f"      {f['slot']} ({f['type']}): {f['status']} (신뢰도 {f['confidence']})")
    print(f"\n  - 보고서({result['report_source']}):\n    {outputs['report']['report_text']}")


if __name__ == "__main__":
    main()
