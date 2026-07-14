# -*- coding: utf-8 -*-
"""
test_core.py
============
핵심 알고리즘에 대한 단위 테스트.
실행: python -m pytest tests/test_core.py -v   (또는 python tests/test_core.py)
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import field_config as fc
import runway_analysis as rwa
import facility_analysis as fca
from geo_dedup import dedup_by_world_distance
from validator import validate_all
from detection import classify_blob, aggregate_temporal_status
import schemas
import tiling
from report_generator import generate_report_offline_template, _enforce_length
from transmitter import transmit_all


class TestRunwayLongestRun(unittest.TestCase):
    def test_middle_blocked(self):
        """RW-03, RW-07이 막혔을 때 최장 가용 구간이 3칸(RW-04~06)인지 확인"""
        order = fc.RUNWAY_SEGMENT_ORDER
        blocked = {"RW-03", "RW-07"}
        run = rwa.longest_available_run(order, blocked)
        self.assertEqual(len(run), 3)
        self.assertIn("RW-04", run)

    def test_no_obstacles(self):
        """장애물이 없으면 전체 10칸이 가용해야 함"""
        order = fc.RUNWAY_SEGMENT_ORDER
        run = rwa.longest_available_run(order, set())
        self.assertEqual(len(run), 10)

    def test_all_blocked(self):
        """전체가 막히면 가용 구간이 0칸이어야 함"""
        order = fc.RUNWAY_SEGMENT_ORDER
        run = rwa.longest_available_run(order, set(order))
        self.assertEqual(len(run), 0)

    def test_length_conversion(self):
        """RW-01 실제 길이가 정확히 300m로 환산되는지 확인 (슬라이드 예시 기준)"""
        length_m = rwa.run_length_meters(["RW-01"])
        self.assertAlmostEqual(length_m, 300.0, places=1)


class TestZoneTileOrder(unittest.TestCase):
    def test_zone_tile_order_matches_runway_and_taxiways(self):
        """ZONE_TILE_ORDER는 활주로+유도로A+유도로B 20개 zone으로 구성되어야 함"""
        self.assertEqual(len(fc.ZONE_TILE_ORDER), 20)
        self.assertEqual(
            set(fc.ZONE_TILE_ORDER),
            set(fc.RUNWAY_SEGMENT_ORDER) | set(fc.TAXIWAY_A_ORDER) | set(fc.TAXIWAY_B_ORDER),
        )


class TestTiling(unittest.TestCase):
    def test_crop_zone_tiles_covers_all_zones_without_overlap(self):
        """탑뷰 캔버스에서 20개 zone 타일이 모두 잘리고, 크기가 실좌표 폭*px_per_cm와 일치해야 함"""
        px_per_cm = fc.WARP_PX_PER_CM
        canvas = np.zeros((fc.WARP_CANVAS_HEIGHT_PX, fc.WARP_CANVAS_WIDTH_PX, 3), dtype=np.uint8)
        tiles = tiling.crop_zone_tiles(canvas, px_per_cm)
        self.assertEqual(set(tiles.keys()), set(fc.ZONE_TILE_ORDER))
        b = fc.SEGMENTS["RW-01"]
        expected_w = int(round((b["x_max"] - b["x_min"]) * px_per_cm))
        expected_h = int(round((b["y_max"] - b["y_min"]) * px_per_cm))
        tile_h, tile_w = tiles["RW-01"].shape[:2]
        self.assertEqual(tile_w, expected_w)
        self.assertEqual(tile_h, expected_h)

    def test_crop_facility_rois_covers_all_slots(self):
        px_per_cm = fc.WARP_PX_PER_CM
        canvas = np.zeros((fc.WARP_CANVAS_HEIGHT_PX, fc.WARP_CANVAS_WIDTH_PX, 3), dtype=np.uint8)
        rois = tiling.crop_facility_rois(canvas, px_per_cm)
        self.assertEqual(set(rois.keys()), set(fc.FACILITY_SLOTS))


class TestFacilitySlotSafetyNet(unittest.TestCase):
    def test_missing_detections_still_fill_6_slots(self):
        """
        핵심 안전장치 테스트: 탐지 결과가 하나도 없어도(빈 딕셔너리)
        보고서에는 반드시 6개 시설물 슬롯이 모두 존재해야 함 ('unconfirmed' 상태로).
        """
        facilities = fca.build_facility_report({})  # 탐지 결과 없음
        self.assertEqual(len(facilities), 6)
        for f in facilities:
            self.assertEqual(f["status"], "unconfirmed")
            self.assertIn(f["slot"], fc.FACILITY_SLOTS)

    def test_partial_detection_fills_rest_as_unconfirmed(self):
        """일부만 탐지되어도 나머지가 'unconfirmed'로 채워져 절대 슬롯이 빠지지 않아야 함"""
        partial = {"FA-01": {"status": "normal", "confidence": 0.9}}
        facilities = fca.build_facility_report(partial)
        self.assertEqual(len(facilities), 6)
        statuses = {f["slot"]: f["status"] for f in facilities}
        self.assertEqual(statuses["FA-01"], "normal")
        self.assertEqual(statuses["FA-02"], "unconfirmed")


class TestAggregateTemporalStatus(unittest.TestCase):
    def test_majority_vote(self):
        results = [("normal", 0.9), ("normal", 0.8), ("fire", 0.4)]
        status, conf = aggregate_temporal_status(results)
        self.assertEqual(status, "normal")
        self.assertAlmostEqual(conf, 0.85, places=2)

    def test_empty_returns_unconfirmed(self):
        status, conf = aggregate_temporal_status([])
        self.assertEqual(status, "unconfirmed")
        self.assertEqual(conf, 0.0)


class TestGeoDedup(unittest.TestCase):
    def test_close_detections_merged(self):
        """5cm 이내의 탐지는 하나로 합쳐져야 함 (중복 카운팅 방지)"""
        dets = [
            {"world_xy": (100.0, 100.0), "confidence": 0.8},
            {"world_xy": (101.0, 100.5), "confidence": 0.9},  # 매우 가까움 -> 병합돼야 함
            {"world_xy": (300.0, 300.0), "confidence": 0.7},  # 멀리 있음 -> 별개
        ]
        merged = dedup_by_world_distance(dets, distance_threshold_cm=5.0)
        self.assertEqual(len(merged), 2)

    def test_empty_input(self):
        self.assertEqual(dedup_by_world_distance([], 5.0), [])


class TestBlobClassification(unittest.TestCase):
    def test_large_round_object_is_crater(self):
        """치수표상 big 폭파구에 가까운 크기+형태는 'crater'로 분류되어야 함"""
        category, subtype, conf = classify_blob(diameter_mm=190, long_axis_mm=200, aspect_ratio=1.1)
        self.assertEqual(category, "crater")

    def test_small_elongated_object_is_uxo(self):
        """작고 매우 길쭉한 물체는 'uxo'(missile 등)으로 분류되어야 함"""
        category, subtype, conf = classify_blob(diameter_mm=70, long_axis_mm=115, aspect_ratio=2.3)
        self.assertEqual(category, "uxo")
        self.assertEqual(subtype, "missile")


class TestReportLength(unittest.TestCase):
    def test_offline_template_within_length_bounds(self):
        summary = {
            "crater_count_total": 2, "runway_crater_count": 1,
            "runway_available_length_m": 300.0,
            "facility_damage_summary": {"normal": 4, "destroy": 1, "fire": 1, "unconfirmed": 0},
            "uxo_count_total": 1, "uxo_runway_count": 1,
        }
        text = generate_report_offline_template(summary)
        self.assertGreaterEqual(len(text), 1)
        self.assertLessEqual(len(text), fc.REPORT_MAX_CHARS)

    def test_enforce_length_truncates_long_text(self):
        long_text = "가" * 200
        text = _enforce_length(long_text)
        self.assertLessEqual(len(text), fc.REPORT_MAX_CHARS)


class TestValidator(unittest.TestCase):
    def test_catches_facility_slot_missing(self):
        """시설물 슬롯이 5개만 있으면(1개 누락) 검증에서 오류로 잡혀야 함"""
        facilities = fca.build_facility_report({})[:5]  # 강제로 1개 슬롯 제거
        facility_list = [{"zone": f["slot"], "status": f["status"]} for f in facilities]
        outputs = {
            "facility_status": schemas.build_facility_status_json("TEST0000", facility_list)
        }
        result = validate_all(outputs)
        self.assertFalse(result["ok"])
        self.assertTrue(any("누락" in e for e in result["errors"]))

    def test_valid_output_passes(self):
        facilities = fca.build_facility_report({})
        facility_list = [{"zone": f["slot"], "status": f["status"]} for f in facilities]
        outputs = {
            "facility_status": schemas.build_facility_status_json(fc.MISSION_CODE, facility_list),
            "crater_detect": schemas.build_crater_detect_json(fc.MISSION_CODE, []),
            "crater_count": schemas.build_crater_count_json(fc.MISSION_CODE, 0),
            "runway_status": schemas.build_runway_status_json(fc.MISSION_CODE, 300000),
            "uxo_detect": schemas.build_uxo_detect_json(fc.MISSION_CODE, []),
            "uxo_count": schemas.build_uxo_count_json(fc.MISSION_CODE, 0),
            "report": schemas.build_report_json(
                fc.MISSION_CODE, "활주로 가용길이 3000.0m로 이착륙 가능, 폭파구 0개, 시설물 전원 정상 확인됨."
            ),
        }
        result = validate_all(outputs)
        self.assertTrue(result["ok"], msg=result["errors"])


class TestTransmitterStub(unittest.TestCase):
    def test_stub_when_endpoint_not_configured(self):
        """엔드포인트 미설정 시 실제 네트워크 호출 없이 스텁 결과만 반환해야 함"""
        outputs = {"start": {"mission_code": "TEST"}}
        result = transmit_all(outputs, endpoint=None)
        self.assertFalse(result["endpoint_configured"])
        self.assertEqual(result["results"][0]["sent"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
