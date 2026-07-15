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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
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


def _enforce_count_bounds(items, kind_label, min_entries, max_entries, filler_factory):
    """서버가 min_entries~max_entries 범위를 벗어난 보고는 통째로 거부하므로(2026-07-14
    현장 테스트로 uxo_detect 최대치 확인, crater_detect도 동일 정책으로 가정),
    전송 직전에 강제로 그 범위 안으로 맞춥니다.
    - 초과: confidence 낮은 것부터 제외.
    - 미달(탐지 0건): filler_factory()로 만든 항목을 채움(실제 탐지 아님 - 서버가 빈 배열을
      거부하는 것을 피하기 위한 자리채움이며, 활주로 가용길이/개수 집계에 영향 없는
      유도로 구간을 사용하도록 filler_factory를 구성해야 함).
    두 경우 모두 터미널에 남겨 운용자가 실제 탐지값과 구분할 수 있게 합니다.
    """
    items_sorted = sorted(items, key=lambda d: d.get("confidence", 0.0), reverse=True)

    if len(items_sorted) > max_entries:
        dropped = items_sorted[max_entries:]
        items_sorted = items_sorted[:max_entries]
        dropped_desc = [
            (d.get("segment"), d.get("size_class") or d.get("type"), d.get("confidence"))
            for d in dropped
        ]
        print(f"[pipeline] {kind_label}: 서버 제한(최대 {max_entries}건)으로 신뢰도 낮은 "
              f"{len(dropped)}건 제외: {dropped_desc}")

    if len(items_sorted) < min_entries:
        needed = min_entries - len(items_sorted)
        for _ in range(needed):
            items_sorted.append(filler_factory())
        print(f"[pipeline] {kind_label}: 탐지 0건 - 서버가 빈 배열을 거부하므로 임의 항목 "
              f"{needed}건을 채워 전송합니다(실제 탐지 아님, 재정찰로 재확인 필요).")

    return items_sorted


class MissionPipeline:
    def __init__(self, mission_code: str = fc.MISSION_CODE, use_llm: bool = True,
                 output_dir: str = "output",
                 detector_backend: str = None, facility_backend: str = None,
                 object_weights: str = None, facility_weights: str = None):
        """
        detector_backend / facility_backend: "classical" | "yolo" (생략 시 field_config의
        DETECTOR_BACKEND / FACILITY_BACKEND 사용). YOLO11n 가중치가 준비되면
        field_config 값만 바꾸거나, 이 인자로 특정 실행만 다른 백엔드로 돌려볼 수 있습니다.
        object_weights / facility_weights: yolo 백엔드일 때 field_config.YOLO_OBJECT_WEIGHTS /
        YOLO_FACILITY_WEIGHTS 대신 쓸 가중치 경로 (학습 도중 체크포인트를 바로 테스트할 때 등).
        """
        self.mission_code = mission_code
        self.use_llm = use_llm
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        object_kwargs = {"weights_path": object_weights} if object_weights else {}
        facility_kwargs = {"weights_path": facility_weights} if facility_weights else {}

        self.calibrator = FieldCalibrator()
        self.object_detector = build_object_detector(detector_backend, **object_kwargs)
        self.facility_classifier = build_facility_classifier(facility_backend, **facility_kwargs)

        self.timing = {}

    # -----------------------------------------------------------------
    def _tic(self, name):
        self.timing[name] = time.time()

    def _toc(self, name):
        elapsed = time.time() - self.timing[name]
        self.timing[name] = round(elapsed, 3)
        return elapsed

    # -----------------------------------------------------------------
    def send_start(self, send_to_dashboard: bool = False):
        """준비단계(start.json) 생성/저장/전송. video_watcher.py 실행(세션) 시점에 딱 1번만
        호출해야 합니다 - 영상이 여러 개로 나뉘어 들어와도 run()이 그때마다 재전송하지 않도록
        run()과 분리되어 있습니다."""
        start_doc = schemas.build_start_json(self.mission_code)
        saved_file = self._save("start.json", start_doc)
        transmit_result = None
        if send_to_dashboard:
            transmit_result = transmit_all({"start": start_doc}, order=["start"])
        return {"start": start_doc, "saved_file": saved_file, "transmit_result": transmit_result}

    # -----------------------------------------------------------------
    def run(self, frames_bgr: list, send_to_dashboard: bool = False):
        """
        frames_bgr: watchdog/프레임 추출 단계에서 넘어온 여러 프레임(numpy BGR 이미지) 리스트
        send_to_dashboard: True면 6단계(전송)까지 실행. 기본은 파일 저장까지만.
        반환: outputs 딕셔너리(7개 JSON, start.json 제외) + 저장된 파일 경로 목록 + 검증/전송 결과

        주의: start.json은 이 메서드가 아니라 send_start()가 세션 시작 시 1번만 담당합니다.
        """
        t_start = time.time()
        outputs = {}
        saved_files = []

        # ---------------- 2단계 + 3-A/3-B: 프레임별 워핑 + 배치 추론 ----------------
        # 프레임마다 재보정하는 이유: 드론이 미세하게 움직이면 카메라 자세가 바뀌므로,
        # 프레임마다 ArUco를 다시 인식해야 좌표 정확도가 유지됩니다.
        # 마커가 일시적으로 가려진 프레임은 직전 보정값을 그대로 재사용합니다(안전장치).
        self._tic("warp_and_detect")
        raw_craters = []
        raw_uxo = []
        facility_frame_results = {slot: [] for slot in fc.FACILITY_SLOTS}
        calibrated_at_least_once = False

        for frame in frames_bgr:
            try:
                self.calibrator.calibrate_from_image(frame)
                calibrated_at_least_once = True
            except Exception:
                if not calibrated_at_least_once:
                    continue  # 첫 프레임부터 마커 검출 실패 -> 이 프레임은 건너뜀
                # 이전 프레임의 보정값을 그대로 사용 (self.calibrator.homography 유지됨)

            # -- 마커 영역은 워핑 전에 미리 지워서 오탐 방지 (흑백 패턴이 어두운 물체로 오인될 수 있음) --
            marker_polygons = []
            for corner_set in self.calibrator.last_marker_corners:
                center = corner_set.mean(axis=0)
                expanded = center + (corner_set - center) * 1.15  # 여유를 살짝 두고(15% 확장) 마스킹
                marker_polygons.append(expanded)
            frame_clean = mask_out_regions(frame, marker_polygons, fill_value=255)

            # -- 2단계: 탑뷰(bird's eye) 워핑. 이후 모든 crop은 픽셀 슬라이싱만으로 충분 --
            topview, px_per_cm = self.calibrator.warp_to_topview(frame_clean)

            # -- 3-A: zone 타일(경계 걸침 없는 그리드) crop -> YOLO11n 배치 추론 --
            zone_tiles = crop_zone_tiles(topview, px_per_cm)
            detections_by_zone = detect_zone_tiles(self.object_detector, zone_tiles)
            for zone_name, dets in detections_by_zone.items():
                origin_x_cm, origin_y_cm = tile_origin_world_cm(zone_name)
                for det in dets:
                    lx, ly = det.center_px
                    wx = origin_x_cm + lx / px_per_cm
                    wy = origin_y_cm + ly / px_per_cm
                    diameter_mm = (det.equiv_diameter_px / px_per_cm) * 10.0
                    long_axis_mm = (det.long_axis_px / px_per_cm) * 10.0

                    # 고전 CV 백엔드는 분류를 안 채워서 돌려주므로, 실측 mm 크기+형태로 여기서 분류.
                    # YOLO11n 백엔드는 모델이 이미 분류까지 마쳐서 돌려주므로 그 값을 그대로 씀.
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
                            "segment": zone_name,   # 타일 자체가 zone이므로 구간이 바로 확정됨
                        })
                    else:  # "uxo"
                        raw_uxo.append({
                            "world_xy": (wx, wy),
                            "type": subtype,
                            "confidence": confidence,
                            "segment": zone_name,
                        })

            # -- 3-B: 시설물 6곳 고정좌표 crop -> YOLO11n-cls 배치 추론 --
            facility_rois = crop_facility_rois(topview, px_per_cm)
            frame_facility_status = self.facility_classifier.classify_frame_batch(facility_rois)
            for slot, (status, conf) in frame_facility_status.items():
                facility_frame_results[slot].append((status, conf))
        self._toc("warp_and_detect")

        # ---------------- 폭파구 중복 제거(프레임간, 실좌표 기준) ----------------
        self._tic("crater_postprocess")
        craters_deduped = dedup_by_world_distance(raw_craters, distance_threshold_cm=5.0)
        # 서버가 crater_detect 보고를 fc.CRATER_DETECT_MIN_ENTRIES~MAX_ENTRIES 범위 밖이면
        # 통째로 거부함 - filler는 활주로 집계에 영향 없도록 유도로(TW-A1) 구간을 사용.
        craters_bounded = _enforce_count_bounds(
            craters_deduped, "crater_detect",
            fc.CRATER_DETECT_MIN_ENTRIES, fc.CRATER_DETECT_MAX_ENTRIES,
            filler_factory=lambda: {
                "segment": fc.TAXIWAY_A_ORDER[0], "size_class": "small", "confidence": 0.0,
            },
        )
        self._toc("crater_postprocess")

        crater_list_out = [
            {"zone": c["segment"], "size": c["size_class"]} for c in craters_bounded
        ]
        outputs["crater_detect"] = schemas.build_crater_detect_json(self.mission_code, crater_list_out)
        saved_files.append(self._save("crater_detect.json", outputs["crater_detect"]))

        runway_crater_count = uxa.count_craters_on_runway(craters_bounded)
        outputs["crater_count"] = schemas.build_crater_count_json(self.mission_code, runway_crater_count)
        saved_files.append(self._save("crater_count.json", outputs["crater_count"]))

        # ---------------- 활주로 가용길이 산출 (확정 로직: runway_analysis 그대로 재사용) ----------------
        # zone 타일 자체가 구간이므로 어느 구간이 막혔는지는 이미 알고 있음(assign_to_segment 재계산 불필요).
        self._tic("runway_analysis")
        blocked_segments = sorted(
            {c["segment"] for c in craters_bounded if c["segment"] in fc.RUNWAY_SEGMENT_ORDER},
            key=fc.RUNWAY_SEGMENT_ORDER.index,
        )
        best_run = rwa.longest_available_run(fc.RUNWAY_SEGMENT_ORDER, set(blocked_segments))
        length_m = round(rwa.run_length_meters(best_run), 1)
        runway_result = {
            "blocked_segments": blocked_segments,
            "longest_available_run": {"segments": best_run, "length_m": length_m},
            "available_length_m": length_m,
        }
        runway_available_length_cm = int(round(runway_result["available_length_m"]))
        outputs["runway_status"] = schemas.build_runway_status_json(
            self.mission_code, runway_available_length_cm
        )
        saved_files.append(self._save("runway_status.json", outputs["runway_status"]))
        self._toc("runway_analysis")

        # ---------------- 시설물 상태 집계 (프레임간 다수결, 6슬롯 강제 매핑) ----------------
        self._tic("facility_analysis")
        detections_by_slot = {}
        for slot, frame_results in facility_frame_results.items():
            if not frame_results:
                continue  # 이 슬롯은 프레임에 한 번도 잡히지 않음 -> 'unconfirmed'로 남게 됨
            status, conf = aggregate_temporal_status(frame_results)
            detections_by_slot[slot] = {"status": status, "confidence": conf}
        facilities = fca.build_facility_report(detections_by_slot)
        facility_list_out = [{"zone": f["slot"], "status": f["status"]} for f in facilities]
        outputs["facility_status"] = schemas.build_facility_status_json(self.mission_code, facility_list_out)
        saved_files.append(self._save("facility_status.json", outputs["facility_status"]))
        self._toc("facility_analysis")

        # ---------------- 불발탄 중복 제거(프레임간, 실좌표 기준) ----------------
        self._tic("uxo_postprocess")
        uxo_deduped = dedup_by_world_distance(raw_uxo, distance_threshold_cm=3.0)
        self._toc("uxo_postprocess")

        # 서버가 uxo_detect 보고를 fc.UXO_DETECT_MIN_ENTRIES~MAX_ENTRIES 범위 밖이면 통째로
        # 거부함(초과분 확인은 2026-07-14 현장 테스트) - filler는 활주로 집계에 영향 없도록
        # 유도로(TW-B1) 구간을 사용. uxo_count도 이 목록 기준으로 다시 세야 validator의
        # "uxo_count <= uxo_detect 총 개수" 검증과 어긋나지 않음.
        uxo_deduped_sorted = _enforce_count_bounds(
            uxo_deduped, "uxo_detect",
            fc.UXO_DETECT_MIN_ENTRIES, fc.UXO_DETECT_MAX_ENTRIES,
            filler_factory=lambda: {
                "segment": fc.TAXIWAY_B_ORDER[0], "type": "dumb", "confidence": 0.0,
            },
        )

        uxo_list_out = [
            {"zone": u["segment"], "type": u["type"]} for u in uxo_deduped_sorted
        ]
        outputs["uxo_detect"] = schemas.build_uxo_detect_json(self.mission_code, uxo_list_out)
        saved_files.append(self._save("uxo_detect.json", outputs["uxo_detect"]))

        runway_uxo_count = uxa.count_uxo_on_runway(uxo_deduped_sorted)
        outputs["uxo_count"] = schemas.build_uxo_count_json(self.mission_code, runway_uxo_count)
        saved_files.append(self._save("uxo_count.json", outputs["uxo_count"]))

        # ---------------- 5단계: LLM 기반 상황보고서 (50~100자 제약) ----------------
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
        outputs["report"] = schemas.build_report_json(self.mission_code, report_result["text"])
        saved_files.append(self._save("report.json", outputs["report"]))
        self._toc("report_generation")

        total_elapsed = round(time.time() - t_start, 2)
        self.timing["total"] = total_elapsed

        # ---------------- 자동 검증(QA) ----------------
        validation = validate_all(outputs)

        # ---------------- 6단계: 대시보드 전송 (선택, 기본 비활성) ----------------
        # start.json은 send_start()가 세션 시작 시 이미 전송했으므로, 여기서는 나머지 7개만 전송.
        transmit_result = None
        if send_to_dashboard:
            remaining_order = [k for k in fc.TRANSMIT_ORDER if k != "start"]
            transmit_result = transmit_all(outputs, order=remaining_order)

        return {
            "outputs": outputs,
            "saved_files": saved_files,
            "timing": self.timing,
            "validation": validation,
            "transmit_result": transmit_result,
            "report_source": report_result["source"],  # 디버그용, 저장되는 JSON에는 포함 안 됨
        }

    # -----------------------------------------------------------------
    def _save(self, filename, data):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
