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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import field_config as fc
import runway_analysis as rwa
import facility_analysis as fca
from geo_dedup import dedup_by_world_distance
from validator import validate_all
from detection import classify_blob
import schemas


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


class TestFacilitySlotSafetyNet(unittest.TestCase):
    def test_missing_detections_still_fill_6_slots(self):
        """
        핵심 안전장치 테스트: 탐지 결과가 하나도 없어도(빈 딕셔너리)
        보고서에는 반드시 6개 시설물 슬롯이 모두 존재해야 함 ('미확인' 상태로).
        """
        facilities = fca.build_facility_report({})  # 탐지 결과 없음
        self.assertEqual(len(facilities), 6)
        for f in facilities:
            self.assertEqual(f["status"], "미확인")
            self.assertIn(f["slot"], fc.FACILITY_SLOTS)

    def test_partial_detection_fills_rest_as_unconfirmed(self):
        """일부만 탐지되어도 나머지가 '미확인'으로 채워져 절대 슬롯이 빠지지 않아야 함"""
        partial = {"FA-01": {"status": "정상", "confidence": 0.9}}
        facilities = fca.build_facility_report(partial)
        self.assertEqual(len(facilities), 6)
        statuses = {f["slot"]: f["status"] for f in facilities}
        self.assertEqual(statuses["FA-01"], "정상")
        self.assertEqual(statuses["FA-02"], "미확인")


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
        """치수표상 대형 폭파구에 가까운 크기+형태는 '폭파구'로 분류되어야 함"""
        category, subtype, conf = classify_blob(diameter_mm=190, long_axis_mm=200, aspect_ratio=1.1)
        self.assertEqual(category, "crater")

    def test_small_elongated_object_is_uxo(self):
        """작고 매우 길쭉한 물체는 '불발탄'(미사일 등)으로 분류되어야 함"""
        category, subtype, conf = classify_blob(diameter_mm=70, long_axis_mm=115, aspect_ratio=2.3)
        self.assertEqual(category, "uxo")
        self.assertEqual(subtype, "미사일")


class TestValidator(unittest.TestCase):
    def test_catches_facility_slot_missing(self):
        """시설물 슬롯이 5개만 있으면(1개 누락) 검증에서 오류로 잡혀야 함"""
        facilities = fca.build_facility_report({})[:5]  # 강제로 1개 슬롯 제거
        outputs = {
            "facility_status": schemas.build_facility_status_json("TEST0000", facilities)
        }
        result = validate_all(outputs)
        self.assertFalse(result["ok"])
        self.assertTrue(any("누락" in e for e in result["errors"]))

    def test_valid_output_passes(self):
        facilities = fca.build_facility_report({})
        outputs = {
            "facility_status": schemas.build_facility_status_json(fc.MISSION_CODE, facilities),
            "crater_detect": schemas.build_crater_detect_json(fc.MISSION_CODE, []),
            "crater_count": schemas.build_crater_count_json(fc.MISSION_CODE, 0),
            "runway_status": schemas.build_runway_status_json(
                fc.MISSION_CODE, {"segments": fc.RUNWAY_SEGMENT_ORDER, "length_m": 3000.0}, [], 3000.0
            ),
            "uxo_detect": schemas.build_uxo_detect_json(fc.MISSION_CODE, []),
            "uxo_count": schemas.build_uxo_count_json(fc.MISSION_CODE, 0),
            "report": schemas.build_report_json(fc.MISSION_CODE, "정상 상황입니다.", {}),
        }
        result = validate_all(outputs)
        self.assertTrue(result["ok"], msg=result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
