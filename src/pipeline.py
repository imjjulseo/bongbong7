# -*- coding: utf-8 -*-
"""
pipeline.py
===========
전체 임무 파이프라인을 순서대로 실행합니다.
(이미지 촬영은 이미 완료되어 프레임 리스트로 주어졌다고 가정 - 드론 제어는 별도 모듈 영역)

흐름:
  1. ArUco 캘리브레이션 (첫 프레임 또는 캐시된 값 사용)
  2. 프레임별 폭파구/불발탄 탐지 -> 실좌표 변환
  3. 지오레퍼런스드 중복 제거
  4. 활주로/유도로 구간 판정 + 최장 가용구간 산출
  5. 시설물 ROI 역투영 + 상태 분류 (6슬롯 강제 매핑)
  6. 로컬 LLM(or 폴백) 보고서 생성
  7. 8개 JSON 파일 생성 + 자동 검증 + 저장(=전송 시뮬레이션)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc
import schemas
from calibration import FieldCalibrator
from detection import classify_blob, mask_out_regions, build_object_detector, build_facility_classifier
from geo_dedup import dedup_by_world_distance
import runway_analysis as rwa
import facility_analysis as fca
import uxo_analysis as uxa
from report_generator import generate_report
from validator import validate_all


class MissionPipeline:
    def __init__(self, mission_code: str = fc.MISSION_CODE, use_llm: bool = True,
                 output_dir: str = "output",
                 detector_backend: str = None, facility_backend: str = None):
        """
        detector_backend / facility_backend: "classical" | "yolo" (생략 시 field_config의
        DETECTOR_BACKEND / FACILITY_BACKEND 사용). 대회 현장에서 YOLO 학습이 끝나면
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
    def run(self, frames_bgr: list):
        """
        frames_bgr: 드론이 촬영한 여러 프레임(numpy BGR 이미지) 리스트
        반환: outputs 딕셔너리(8개 JSON) + 저장된 파일 경로 목록
        """
        t_start = time.time()
        outputs = {}
        saved_files = []

        # ---------------- 1. 준비단계 (start.json) ----------------
        outputs["start"] = schemas.build_start_json(self.mission_code)
        saved_files.append(self._save("start.json", outputs["start"]))

        # ---------------- 2~6. 프레임별 처리 (한 번의 루프로 통합) ----------------
        # 프레임마다 재보정하는 이유: 드론이 미세하게 움직이면 카메라 자세가 바뀌므로,
        # 프레임마다 ArUco를 다시 인식해야 좌표 정확도가 유지됩니다.
        # 마커가 일시적으로 가려진 프레임은 직전 보정값을 그대로 재사용합니다(안전장치).
        self._tic("calibration_and_detection")
        raw_craters = []
        raw_uxo = []
        facility_rois = {slot: [] for slot in fc.FACILITY_SLOTS}
        calibrated_at_least_once = False

        for frame in frames_bgr:
            try:
                self.calibrator.calibrate_from_image(frame)
                calibrated_at_least_once = True
            except Exception:
                if not calibrated_at_least_once:
                    continue  # 첫 프레임부터 마커 검출 실패 -> 이 프레임은 건너뜀
                # 이전 프레임의 보정값을 그대로 사용 (self.calibrator.homography 유지됨)

            # -- 마커 영역은 미리 지워서 오탐 방지 (흑백 패턴이 어두운 물체로 오인될 수 있음) --
            marker_polygons = []
            for corner_set in self.calibrator.last_marker_corners:
                # 여유를 살짝 두고(10% 확장) 마스킹해 경계 부분까지 확실히 제거
                center = corner_set.mean(axis=0)
                expanded = center + (corner_set - center) * 1.15
                marker_polygons.append(expanded)
            frame_clean = mask_out_regions(frame, marker_polygons, fill_value=255)

            # -- 폭파구/불발탄 통합 탐지 (백엔드는 field_config.DETECTOR_BACKEND로 전환) --
            # 고전 CV 백엔드는 분류를 안 채워서 돌려주므로, 실측 mm 크기+형태로 여기서 분류합니다.
            # YOLO 백엔드는 모델이 이미 분류까지 마쳐서 돌려주므로 그 값을 그대로 씁니다.
            for det in self.object_detector.detect(frame_clean):
                px, py = det.center_px
                wx, wy = self.calibrator.pixel_to_world(px, py)
                diameter_cm = self._pixel_length_to_world_cm(px, py, det.equiv_diameter_px)
                diameter_mm = diameter_cm * 10.0
                long_axis_cm = self._pixel_length_to_world_cm(px, py, det.long_axis_px)
                long_axis_mm = long_axis_cm * 10.0

                if det.category is not None:
                    category, subtype, confidence = det.category, det.subtype, det.confidence
                else:
                    category, subtype, confidence = classify_blob(diameter_mm, long_axis_mm, det.aspect_ratio)

                if category == "crater":
                    raw_craters.append({
                        "world_xy": (wx, wy),
                        "diameter_mm": round(diameter_mm, 1),
                        "size_class": subtype,
                        "confidence": confidence,
                    })
                else:  # "uxo"
                    raw_uxo.append({
                        "world_xy": (wx, wy),
                        "type": subtype,
                        "confidence": confidence,
                    })

            # -- 시설물 ROI 크롭 (이 프레임의 호모그래피 기준으로 역투영) --
            for slot in fc.FACILITY_SLOTS:
                b = fc.SEGMENTS[slot]
                try:
                    x, y, w, h = self.calibrator.world_bbox_to_pixel_bbox(
                        b["x_min"], b["y_min"], b["x_max"], b["y_max"], frame.shape
                    )
                    roi = frame[y:y + h, x:x + w]
                    if roi.size > 0:
                        facility_rois[slot].append(roi)
                except Exception:
                    continue
        self._toc("calibration_and_detection")

        # ---------------- 3. 폭파구 중복 제거 + 구간 배정 ----------------
        self._tic("crater_postprocess")
        craters_deduped = dedup_by_world_distance(raw_craters, distance_threshold_cm=5.0)
        craters_with_seg = uxa.assign_crater_segments(craters_deduped)
        self._toc("crater_postprocess")

        crater_list_out = []
        for i, c in enumerate(craters_with_seg):
            crater_list_out.append({
                "id": f"CR-{i+1:03d}",
                "segment": c["segment"],
                "size_class": c["size_class"],
                "center_world_cm": [round(c["world_xy"][0], 1), round(c["world_xy"][1], 1)],
                "diameter_m": round(c["diameter_mm"] / 1000.0, 2),
            })
        outputs["crater_detect"] = schemas.build_crater_detect_json(self.mission_code, crater_list_out)
        saved_files.append(self._save("crater_detect.json", outputs["crater_detect"]))

        runway_crater_count = uxa.count_craters_on_runway(craters_with_seg)
        outputs["crater_count"] = schemas.build_crater_count_json(self.mission_code, runway_crater_count)
        saved_files.append(self._save("crater_count.json", outputs["crater_count"]))

        # ---------------- 4. 활주로 가용길이 산출 ----------------
        self._tic("runway_analysis")
        runway_obstacle_points = [c["world_xy"] for c in craters_with_seg
                                   if c["segment"] in fc.RUNWAY_SEGMENT_ORDER]
        runway_result = rwa.analyze_runway(runway_obstacle_points)
        outputs["runway_status"] = schemas.build_runway_status_json(
            self.mission_code,
            runway_result["longest_available_run"],
            runway_result["blocked_segments"],
            runway_result["available_length_m"],
        )
        saved_files.append(self._save("runway_status.json", outputs["runway_status"]))
        self._toc("runway_analysis")

        # ---------------- 5. 시설물 상태 분류 (6슬롯 강제 매핑) ----------------
        self._tic("facility_analysis")
        detections_by_slot = {}
        for slot, rois in facility_rois.items():
            if not rois:
                continue  # 이 슬롯은 프레임에 한 번도 잡히지 않음 -> '미확인'으로 남게 됨
            status, conf = self.facility_classifier.classify(rois)
            detections_by_slot[slot] = {"status": status, "confidence": conf}
        facilities = fca.build_facility_report(detections_by_slot)
        outputs["facility_status"] = schemas.build_facility_status_json(self.mission_code, facilities)
        saved_files.append(self._save("facility_status.json", outputs["facility_status"]))
        self._toc("facility_analysis")

        # ---------------- 6. 불발탄 중복 제거 + 구간 배정 ----------------
        self._tic("uxo_postprocess")
        uxo_deduped = dedup_by_world_distance(raw_uxo, distance_threshold_cm=3.0)
        uxo_with_seg = uxa.assign_uxo_segments(uxo_deduped)
        self._toc("uxo_postprocess")

        uxo_list_out = []
        for i, u in enumerate(uxo_with_seg):
            uxo_list_out.append({
                "id": f"UXO-{i+1:03d}",
                "segment": u["segment"],
                "type": u["type"],
                "center_world_cm": [round(u["world_xy"][0], 1), round(u["world_xy"][1], 1)],
                "confidence": u["confidence"],
            })
        outputs["uxo_detect"] = schemas.build_uxo_detect_json(self.mission_code, uxo_list_out)
        saved_files.append(self._save("uxo_detect.json", outputs["uxo_detect"]))

        runway_uxo_count = uxa.count_uxo_on_runway(uxo_with_seg)
        outputs["uxo_count"] = schemas.build_uxo_count_json(self.mission_code, runway_uxo_count)
        saved_files.append(self._save("uxo_count.json", outputs["uxo_count"]))

        # ---------------- 7. LLM 기반 상황보고서 ----------------
        self._tic("report_generation")
        damage_summary = fca.summarize_damage(facilities)
        report_summary = {
            "crater_count_total": len(crater_list_out),
            "runway_crater_count": runway_crater_count,
            "runway_available_length_m": runway_result["available_length_m"],
            "runway_blocked_segments": runway_result["blocked_segments"],
            "facility_damage_summary": damage_summary,
            "uxo_count_total": len(uxo_list_out),
            "uxo_runway_count": runway_uxo_count,
        }
        report_result = generate_report(report_summary, use_llm=self.use_llm)
        outputs["report"] = schemas.build_report_json(
            self.mission_code, report_result["text"], report_summary
        )
        saved_files.append(self._save("report.json", outputs["report"]))
        self._toc("report_generation")

        total_elapsed = round(time.time() - t_start, 2)
        self.timing["total"] = total_elapsed

        # ---------------- 8. 자동 검증 ----------------
        validation = validate_all(outputs)

        return {
            "outputs": outputs,
            "saved_files": saved_files,
            "timing": self.timing,
            "validation": validation,
            "report_source": report_result["source"],  # 디버그용, 저장되는 JSON에는 포함 안 됨
        }

    # -----------------------------------------------------------------
    def _pixel_length_to_world_cm(self, px, py, length_px):
        """픽셀 거리 -> 실좌표 거리(cm) 근사 변환 (해당 지점 주변의 국소 스케일 사용)"""
        wx1, wy1 = self.calibrator.pixel_to_world(px, py)
        wx2, wy2 = self.calibrator.pixel_to_world(px + length_px, py)
        return ((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2) ** 0.5

    def _save(self, filename, data):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
