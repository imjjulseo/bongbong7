# -*- coding: utf-8 -*-
"""
runway_analysis.py
===================
활주로 가용길이 = "폭파구/불발탄이 있는 구간을 제외했을 때, 가장 긴 연속 가용 구간의 길이"
(사용자가 명시한 정의: 10개 구간 중 장애물이 있는 구간을 빼고 남은 구간 중 최장 연속 구간)

알고리즘: 각 구간을 막힘(1)/가용(0)으로 표시한 뒤 최장 연속 0 구간을 O(n)으로 탐색.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


def point_in_segment(x_cm, y_cm, seg_bounds):
    return (seg_bounds["x_min"] <= x_cm <= seg_bounds["x_max"] and
            seg_bounds["y_min"] <= y_cm <= seg_bounds["y_max"])


def assign_to_segment(x_cm, y_cm, candidate_order):
    """
    실좌표(x_cm, y_cm)가 어느 구간(RW-01 등)에 속하는지 판정.
    구간 경계에 걸친 경우 '중심좌표가 속한 구간'을 우선하고,
    만약 경계선 위(오차범위 내)라면 더 많이 겹치는 쪽으로 배정합니다.
    """
    for seg_name in candidate_order:
        bounds = fc.SEGMENTS[seg_name]
        if point_in_segment(x_cm, y_cm, bounds):
            return seg_name
    return None  # 해당 구역 밖


def compute_blocked_segments(obstacle_world_points: list, candidate_order: list,
                              obstacle_radius_cm: float = 0.0):
    """
    obstacle_world_points: [(x_cm, y_cm), ...]  (폭파구/불발탄 등 장애물의 실좌표 중심점)
    obstacle_radius_cm: 장애물 반경을 고려해, 경계에 걸친 경우까지 막힘으로 처리하고 싶을 때 사용

    반환: set(막힌 구간 이름들)
    """
    blocked = set()
    for (x, y) in obstacle_world_points:
        seg = assign_to_segment(x, y, candidate_order)
        if seg is not None:
            blocked.add(seg)
            continue
        # 반경을 고려한 근접 판정 (장애물이 경계선에 걸쳐있는 경우)
        if obstacle_radius_cm > 0:
            for seg_name in candidate_order:
                b = fc.SEGMENTS[seg_name]
                nearest_x = min(max(x, b["x_min"]), b["x_max"])
                nearest_y = min(max(y, b["y_min"]), b["y_max"])
                dist = ((x - nearest_x) ** 2 + (y - nearest_y) ** 2) ** 0.5
                if dist <= obstacle_radius_cm:
                    blocked.add(seg_name)
    return blocked


def longest_available_run(candidate_order: list, blocked_segments: set):
    """
    핵심 알고리즘: 막힌 구간을 제외했을 때 가장 긴 '연속' 가용 구간을 찾음.
    예) RW-01~10 중 RW-03, RW-07이 막혔다면
        가용 런: [RW-01,RW-02](2칸), [RW-04,RW-05,RW-06](3칸), [RW-08,RW-09,RW-10](3칸)
        -> 길이가 같은 런이 여럿이면 먼저 나오는 쪽을 반환 (원하면 정책 변경 가능)
    """
    best_run = []
    current_run = []

    def _flush():
        nonlocal best_run, current_run
        if len(current_run) > len(best_run):
            best_run = current_run.copy()
        current_run = []

    for seg_name in candidate_order:
        if seg_name in blocked_segments:
            _flush()
        else:
            current_run.append(seg_name)
    _flush()  # 마지막 런 처리

    return best_run


def run_length_meters(run_segments: list):
    """구간 리스트의 실제 길이(m) 합산 (모형 cm * 6.0 m/cm 환산)"""
    total_cm = 0.0
    for seg_name in run_segments:
        b = fc.SEGMENTS[seg_name]
        total_cm += (b["x_max"] - b["x_min"])
    return total_cm * fc.REAL_METERS_PER_MODEL_CM


def analyze_runway(obstacle_world_points: list, obstacle_radius_cm: float = 0.0):
    """
    활주로(RW-01~10) 전체 분석: 막힌 구간, 최장 가용 런, 가용길이(m)를 한번에 계산.
    """
    order = fc.RUNWAY_SEGMENT_ORDER
    blocked = compute_blocked_segments(obstacle_world_points, order, obstacle_radius_cm)
    best_run = longest_available_run(order, blocked)
    length_m = run_length_meters(best_run)

    return {
        "blocked_segments": sorted(blocked, key=lambda s: order.index(s)),
        "longest_available_run": {"segments": best_run, "length_m": round(length_m, 1)},
        "available_length_m": round(length_m, 1),
    }
