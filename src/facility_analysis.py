# -*- coding: utf-8 -*-
"""
facility_analysis.py
=====================
시설물은 위치가 고정되어 있다는 사전 지식을 활용합니다.
탐지 여부와 상관없이 6개 슬롯(FA-01~06)을 항상 보고서에 채워 넣어
'시설물 누락'으로 인한 감점을 원천 차단합니다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


def build_facility_report(detections_by_slot: dict):
    """
    detections_by_slot: {"FA-01": {"status": "normal", "confidence": 0.9}, ...}
                         탐지 못한 슬롯은 딕셔너리에 아예 없어도 됨.

    반환: 6개 슬롯이 항상 채워진 리스트
          [{"slot":..., "type":..., "status":..., "confidence":...}, ...]
    """
    facilities = []
    for slot in fc.FACILITY_SLOTS:
        facility_type = fc.FACILITY_TYPE_BY_SLOT[slot]
        if slot in detections_by_slot:
            status = detections_by_slot[slot].get("status", "unconfirmed")
            confidence = detections_by_slot[slot].get("confidence", 0.0)
        else:
            # 탐지 실패 -> 슬롯을 비우지 않고 'unconfirmed'로 명시 (누락 방지 핵심)
            status = "unconfirmed"
            confidence = 0.0

        facilities.append({
            "slot": slot,
            "type": facility_type,
            "status": status,
            "confidence": confidence,
        })
    return facilities


def summarize_damage(facilities: list):
    """전체 시설물 중 destroy/fire/unconfirmed 개수 요약 (LLM 보고서 생성용 보조 통계)"""
    counts = {"normal": 0, "destroy": 0, "fire": 0, "unconfirmed": 0}
    for f in facilities:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    return counts
