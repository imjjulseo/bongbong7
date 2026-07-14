# -*- coding: utf-8 -*-
"""
pipeline.py
===========
전체 임무 파이프라인을 순서대로 실행합니다. (확정된 신규 파이프라인 기준)

흐름:
  1. (별도 프로세스) 영상 watchdog 감시 -> 지정 프레임 간격마다 이미지 추출
     scripts/video_watcher.py가 담당하며, 이 모듈(MissionPipeline)은 추출된
     프레임 리스트를 입력으로 받는 지점부터 시작합니다.
  2. 프레임별 ArUco 마커 검출 -> 호모그래피 계산 -> 탑뷰(bird's eye) 워핑
     (이 단계 이후 픽셀좌표 == 실좌표(cm) * px_per_cm 로 정렬됨 -> 프레임별 역투영 불필요)
  3-A. [활주로/유도로] zone별 타일 crop(경계 걸침 없는 단순 그리드, overlap 불필요)
       -> YOLO11n 배치 추론(한 번의 predict 호출) - 폭파구 big/medium/small +
          불발탄 missile/dumb/cluster -> zone별 결과 취합 -> 가용거리 계산(runway_analysis)
  3-B. [시설물] 고정 좌표 6곳(FA-01~06) crop -> YOLO11n-cls 배치 추론
       -> normal/destroy/fire 3클래스 분류 (6슬롯 강제 매핑으로 누락 방지)
  4. 3-A, 3-B 결과 통합 -> mission_code 포함 8종 JSON 매핑
  5. LLM(또는 오프라인 템플릿) 상황보고서 생성 (50~100자 제약)
  6. (선택) 전송 모듈로 대시보드에 순차/중복 전송
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc
import schemas
from calibration import FieldCalibrator
from detection import (
    classify_blob, mask_out_regions, build_object_detector, build_facility_classifier,
    detect_zone_tiles, aggregate_temporal_status,
)
from tiling import crop_zone_tiles, crop_facility_rois, tile_origin_world_cm
from geo_dedup import dedup_by_world_distance
import runway_analysis as rwa
import facility_analysis as fca
import uxo_analysis as uxa
from report_generator import generate_report
from validator import validate_all
from transmitter import transmit_all


class MissionPipeline:
    def __init__(self, mission_code: str = fc.MISSION_CODE, use_llm: bool = True,
                 output_dir: str = "output",
                 detector_backend: str = None, facility_backend: str = None):
        """
        detector_backend / facility_backend: "classical" | "yolo" (생략 시 field_config의
        DETECTOR_BACKEND / FACILITY_BACKEND 사용). YOLO11n 가중치가 준비되면
        field_config 값만 바꾸거나, 이 인자로 특정 실행만 다른 백엔드로 돌려볼 수 있습니다.
        """
        self.mission_code = mission_code
        self.use_llm = use_llm
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.calibrator = FieldCalibrator()
        self.object_detector = build_object_detector(detector_backend)
        self.facility_classifier = build_facility_classifier(facility_backend)

        self.timing = {}

    # -----------------------------------------------------------------
    def _tic(self, name):
        self.timing[name] = time.time()

    def _toc(self, name):
        elapsed = time.time() - self.timing[name]
        self.timing[name] = round(elapsed, 3)
        return elapsed

    # -----------------------------------------------------------------
    def run(self, frames_bgr: list, send_to_dashboard: bool = False):
        """
        frames_bgr: watchdog/프레임 추출 단계에서 넘어온 여러 프레임(numpy BGR 이미지) 리스트
        send_to_dashboard: True면 6단계(전송)까지 실행. 기본은 파일 저장까지만.
        반환: outputs 딕셔너리(8개 JSON) + 저장된 파일 경로 목록 + 검증/전송 결과
        """
        t_start = time.time()
        saved_files = []

        # ---------------- 2단계 + 3-A/3-B: 프레임별 워핑 + 배치 추론 ----------------
        # 프레임마다 재보정하는 이유: 드론이 미세하게 움직이면 카메라 자세가 바뀌므로,
        # 프레임마다 ArUco를 다시 인식해야 좌표 정확도가 유지됩니다.
        # 마커가 일시적으로 가려진 프레임은 직전 보정값을 그대로 재사용합니다(안전장치).
        self._tic("warp_and_detect")

        rest_frames = 0

        for frame in frames_bgr:
            try:
                self.calibrator.calibrate_from_image(frame)
            except Exception:
                continue  # 첫 프레임부터 마커 검출 실패 -> 이 프레임은 건너뜀
            
            if rest_frames > 0:
                rest_frames -= 1
                continue
            else:
                rest_frames = 5
            

            # -- 마커 영역은 워핑 전에 미리 지워서 오탐 방지 (흑백 패턴이 어두운 물체로 오인될 수 있음) --
            marker_polygons = []
            for corner_set in self.calibrator.last_marker_corners:
                center = corner_set.mean(axis=0)
                expanded = center + (corner_set - center) * 1.15  # 여유를 살짝 두고(15% 확장) 마스킹
                marker_polygons.append(expanded)
            frame_clean = mask_out_regions(frame, marker_polygons, fill_value=255)

            # -- 2단계: 탑뷰(bird's eye) 워핑. 이후 모든 crop은 픽셀 슬라이싱만으로 충분 --
            topview, px_per_cm = self.calibrator.warp_to_topview(frame_clean)

            saved_files.append((topview, px_per_cm))


        return saved_files

    # -----------------------------------------------------------------
    def _save(self, filename, data):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
