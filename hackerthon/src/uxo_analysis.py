# -*- coding: utf-8 -*-
"""
uxo_analysis.py
================
불발탄 탐지 결과를 구간에 배정하고, 활주로 구간 내 총 개수를 산출합니다.
크레이터와 동일한 구간 판정 로직(runway_analysis.assign_to_segment)을 재사용합니다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc
from runway_analysis import assign_to_segment


ALL_ZONE_ORDER = (fc.RUNWAY_SEGMENT_ORDER + fc.TAXIWAY_A_ORDER +
                   fc.TAXIWAY_B_ORDER + fc.FACILITY_SLOTS)


def assign_uxo_segments(uxo_detections: list):
    """
    uxo_detections: [{"world_xy": (x,y), "type":..., "confidence":...}, ...]
    각 탐지에 "segment" 필드를 채워서 반환.
    """
    out = []
    for det in uxo_detections:
        x, y = det["world_xy"]
        seg = assign_to_segment(x, y, ALL_ZONE_ORDER)
        item = dict(det)
        item["segment"] = seg if seg else "미분류"
        out.append(item)
    return out


def count_uxo_on_runway(uxo_with_segments: list):
    """활주로(RW-01~10) 구간에 위치한 불발탄 개수만 집계"""
    runway_set = set(fc.RUNWAY_SEGMENT_ORDER)
    return sum(1 for d in uxo_with_segments if d.get("segment") in runway_set)


def assign_crater_segments(crater_detections: list):
    """폭파구도 동일한 방식으로 구간을 배정 (활주로+유도로 전체 대상)"""
    out = []
    for det in crater_detections:
        x, y = det["world_xy"]
        seg = assign_to_segment(x, y, ALL_ZONE_ORDER)
        item = dict(det)
        item["segment"] = seg if seg else "미분류"
        out.append(item)
    return out


def count_craters_on_runway(crater_with_segments: list):
    runway_set = set(fc.RUNWAY_SEGMENT_ORDER)
    return sum(1 for d in crater_with_segments if d.get("segment") in runway_set)
