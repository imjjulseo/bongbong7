# -*- coding: utf-8 -*-
"""
generate_test_scene.py
=======================
실제 드론 촬영 사진이 없는 상태에서도 전체 파이프라인이 정상 동작하는지 검증하기 위해,
ArUco 마커 + 경기장 구획 + 미션 객체(폭파구/불발탄/시설물)를 합성한 테스트 이미지를 만듭니다.
대회 슬라이드에 공개된 실제 모형 사진(폭파구/불발탄/시설물 디오라마) 형태를 최대한
흉내 내도록 아이콘/텍스처를 그립니다.

이 스크립트가 만드는 시나리오:
  - RW-03, RW-07 구간에 폭파구 배치 -> 최장 가용구간은 RW-08~RW-10 (3칸=150m) 이어야 함
    (RW-04~06도 3칸이지만 먼저 나오는 순서 정책상 RW-04가 먼저 선택될 수 있음 -> 실행 결과로 확인)
  - RW-05에 자탄(uxo) 배치 -> 활주로 불발탄 개수 1개로 집계되어야 함
  - TW-B2에 포탄 배치 -> 활주로 불발탄 집계에는 포함되지 않아야 함(유도로이므로)
  - FA-02(관제레이더)에 화재 표시, FA-03(격납고)에 파손 표시, 나머지는 정상
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import field_config as fc
from img_io import imwrite_safe

PX_PER_CM = 4
MARGIN_PX = 60


def world_to_canvas(wx, wy):
    return int(MARGIN_PX + wx * PX_PER_CM), int(MARGIN_PX + wy * PX_PER_CM)


def _seeded_rng(*key_parts):
    """구간/좌표 등으로부터 결정론적(재현 가능한) 난수 생성기를 만듦"""
    return np.random.default_rng(abs(hash(key_parts)) % (2 ** 32))


# =====================================================================
# 바닥 텍스처 (활주로/유도로/시설물 구역) - 슬라이드의 경기장 레이아웃 이미지 참고
# =====================================================================
def _draw_runway_ground(canvas, p1, p2, seg_name):
    """활주로: 회색 아스팔트 + 중앙 파선 + 약한 질감 노이즈"""
    cv2.rectangle(canvas, p1, p2, (118, 118, 118), -1)

    rng = _seeded_rng(seg_name, "tex")
    n = max(6, ((p2[0] - p1[0]) * (p2[1] - p1[1])) // 90)
    xs = rng.integers(p1[0], max(p1[0] + 1, p2[0]), size=n)
    ys = rng.integers(p1[1], max(p1[1] + 1, p2[1]), size=n)
    for x, y in zip(xs, ys):
        shade = int(np.clip(118 + rng.integers(-14, 14), 70, 170))
        cv2.circle(canvas, (int(x), int(y)), 1, (shade, shade, shade), -1)

    cy = (p1[1] + p2[1]) // 2
    x = p1[0] + 4
    while x < p2[0] - 4:
        x2 = min(x + 9, p2[0] - 4)
        cv2.line(canvas, (x, cy), (x2, cy), (222, 222, 222), 2)
        x += 9 + 7


def _draw_taxiway_ground(canvas, p1, p2, seg_name):
    """유도로: 녹회색 포장 + 노란 중앙 실선"""
    cv2.rectangle(canvas, p1, p2, (108, 138, 112), -1)

    rng = _seeded_rng(seg_name, "tex")
    n = max(6, ((p2[0] - p1[0]) * (p2[1] - p1[1])) // 110)
    xs = rng.integers(p1[0], max(p1[0] + 1, p2[0]), size=n)
    ys = rng.integers(p1[1], max(p1[1] + 1, p2[1]), size=n)
    for x, y in zip(xs, ys):
        d = int(rng.integers(-10, 10))
        cv2.circle(canvas, (int(x), int(y)),
                   1, (108 + d, 138 + d, 112 + d), -1)

    cy = (p1[1] + p2[1]) // 2
    cv2.line(canvas, (p1[0] + 3, cy), (p2[0] - 3, cy), (0, 200, 220), 1)


def _draw_facility_ground(canvas, p1, p2, seg_name):
    """시설물 구역: 잔디+포장 혼합 바닥"""
    cv2.rectangle(canvas, p1, p2, (150, 176, 150), -1)

    rng = _seeded_rng(seg_name, "tex")
    n = max(10, ((p2[0] - p1[0]) * (p2[1] - p1[1])) // 45)
    xs = rng.integers(p1[0], max(p1[0] + 1, p2[0]), size=n)
    ys = rng.integers(p1[1], max(p1[1] + 1, p2[1]), size=n)
    for x, y in zip(xs, ys):
        cv2.circle(canvas, (int(x), int(y)), 1, (108, 150, 100), -1)


def draw_field_base(canvas):
    """활주로/유도로/시설물 구역을 각기 다른 바닥 텍스처로 그림"""
    for seg_name, b in fc.SEGMENTS.items():
        p1 = world_to_canvas(b["x_min"], b["y_min"])
        p2 = world_to_canvas(b["x_max"], b["y_max"])
        zone = seg_name[:2]
        if zone == "RW":
            _draw_runway_ground(canvas, p1, p2, seg_name)
        elif zone == "TW":
            _draw_taxiway_ground(canvas, p1, p2, seg_name)
        else:
            _draw_facility_ground(canvas, p1, p2, seg_name)
        cv2.rectangle(canvas, p1, p2, (40, 40, 40), thickness=1)
        cv2.putText(canvas, seg_name, (p1[0] + 3, p1[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
    _draw_runway_threshold_marks(canvas)
    return canvas


def _draw_runway_threshold_marks(canvas):
    """활주로 양 끝(이착륙 진입부)에 흰색 임계값 마킹(hash marks)을 그림"""
    first_b = fc.SEGMENTS[fc.RUNWAY_SEGMENT_ORDER[0]]
    last_b = fc.SEGMENTS[fc.RUNWAY_SEGMENT_ORDER[-1]]
    for b, side in ((first_b, "left"), (last_b, "right")):
        p1 = world_to_canvas(b["x_min"], b["y_min"])
        p2 = world_to_canvas(b["x_max"], b["y_max"])
        edge_x = p1[0] + 5 if side == "left" else p2[0] - 5
        span = p2[1] - p1[1]
        for k in range(5):
            y = p1[1] + span * (k + 1) // 6
            cv2.line(canvas, (edge_x - 3, y), (edge_x + 3, y), (228, 228, 228), 2)


def draw_aruco_markers(canvas, aruco_dict):
    """4개 모서리 마커를 grid 좌표에 렌더링 (흰색 여백/quiet zone 포함 - 검출에 필수)"""
    marker_core_px = 50
    quiet_zone_px = 16  # 마커 주변 흰 여백 두께
    block_size = marker_core_px + 2 * quiet_zone_px

    for marker_id, (wx, wy) in fc.ARUCO_MARKER_WORLD_POSITIONS.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_core_px)
        # 흰 배경 블록 위에 마커를 중앙 배치 -> quiet zone 확보
        block = np.full((block_size, block_size), 255, dtype=np.uint8)
        block[quiet_zone_px:quiet_zone_px + marker_core_px,
              quiet_zone_px:quiet_zone_px + marker_core_px] = marker_img
        block_bgr = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)

        cx, cy = world_to_canvas(wx, wy)
        x0, y0 = cx - block_size // 2, cy - block_size // 2
        h, w = canvas.shape[:2]
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x0 + block_size), min(h, y0 + block_size)
        if x1c > x0c and y1c > y0c:
            canvas[y0c:y1c, x0c:x1c] = block_bgr[
                (y0c - y0):(y1c - y0), (x0c - x0):(x1c - x0)
            ]
    return canvas


def draw_crater(canvas, segment_name, size_class="medium"):
    """지정 구간 중앙에 검정 불규칙 blob(폭파구)을 그림 (실측 모형 사진과 이미 유사해 그대로 유지)"""
    b = fc.SEGMENTS[segment_name]
    cx_world = (b["x_min"] + b["x_max"]) / 2
    cy_world = (b["y_min"] + b["y_max"]) / 2
    cx, cy = world_to_canvas(cx_world, cy_world)

    dims = fc.CRATER_SIZE_TABLE_MM[size_class]
    radius_cm = (dims["w"] / 10.0) / 2.0  # mm -> cm -> 반지름
    radius_px = max(6, int(radius_cm * PX_PER_CM))

    # 완전한 원이 아니라 살짝 불규칙한 다각형으로 그려서 '자연스러운' 폭파구 형태 흉내
    rng = _seeded_rng(segment_name)
    angles = np.linspace(0, 2 * np.pi, 14)
    pts = []
    for a in angles:
        r = radius_px * (0.8 + 0.4 * rng.random())
        pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
    cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], (15, 15, 15))
    return cx_world, cy_world


# =====================================================================
# 불발탄 형태 (미사일=원뿔+핀 달린 원통, 포탄=테이퍼진 원통+꼬리핀, 자탄=울퉁불퉁한 구형)
# =====================================================================
def _draw_missile(canvas, cx, cy, w_px, d_px, color=(12, 12, 12)):
    """실측 치수(w_px x d_px) bounding box 안에서만 그려서 분류기의 종횡비 판정이 깨지지 않게 함"""
    half_w = max(2, w_px // 2)
    half_h = max(4, d_px // 2)
    top, bottom = cy - half_h, cy + half_h
    nose_h = min(half_w, 2 * half_h - 2)
    body_top = top + nose_h
    fin_h = min(max(1, half_w // 2), (bottom - body_top) // 2)
    body_bottom = bottom - fin_h

    cv2.rectangle(canvas, (cx - half_w, body_top), (cx + half_w, body_bottom), color, -1)

    nose = np.array([
        [cx, top],
        [cx - half_w, body_top],
        [cx + half_w, body_top],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [nose], color)

    # 꼬리 핀: bounding box 폭(half_w) 안에서만 좁아지는 사다리꼴로 표현
    fin = np.array([
        [cx - half_w, body_bottom],
        [cx - half_w // 2, bottom],
        [cx + half_w // 2, bottom],
        [cx + half_w, body_bottom],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [fin], color)


def _draw_mortar_round(canvas, cx, cy, w_px, d_px, color=(52, 52, 52)):
    """실측 치수(w_px x d_px) bounding box 안에서만 그려서 분류기의 종횡비 판정이 깨지지 않게 함"""
    half_w = max(2, w_px // 2)
    half_h = max(4, d_px // 2)
    top, bottom = cy - half_h, cy + half_h
    taper = max(1, half_w // 3)

    body = np.array([
        [cx - half_w, top + taper],
        [cx - half_w + taper, top],
        [cx + half_w - taper, top],
        [cx + half_w, top + taper],
        [cx + half_w, bottom],
        [cx - half_w, bottom],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [body], color)

    # 꼬리 핀: 몸체보다 살짝 어두운 색의 밑단 띠로 표현 (폭은 몸체와 동일하게 유지)
    fin_h = max(2, half_w // 2)
    fin_color = tuple(max(0, c - 15) for c in color)
    cv2.rectangle(canvas, (cx - half_w, bottom - fin_h), (cx + half_w, bottom), fin_color, -1)


def _draw_submunition(canvas, cx, cy, w_px, d_px, color=(34, 44, 34)):
    r = max(3, min(w_px, d_px) // 2)
    cv2.circle(canvas, (cx, cy), r, color, -1)

    # 표면 굴곡(자탄 특유의 울퉁불퉁한 표면) 표현 - 원 둘레에 작은 돌기 배치
    rng = _seeded_rng(cx, cy)
    n_bumps = 6
    bump_color = tuple(max(0, c - 10) for c in color)
    for i in range(n_bumps):
        angle = 2 * np.pi * i / n_bumps + rng.uniform(-0.2, 0.2)
        bump_r = max(1, r // 3)
        bx = int(cx + (r - bump_r * 0.6) * np.cos(angle))
        by = int(cy + (r - bump_r * 0.6) * np.sin(angle))
        cv2.circle(canvas, (bx, by), bump_r, bump_color, -1)


def draw_uxo(canvas, segment_name, uxo_type="cluster", offset_cm=(0, 0)):
    """지정 구간에 종류별로 구분되는 불발탄 형태를 그림"""
    b = fc.SEGMENTS[segment_name]
    cx_world = (b["x_min"] + b["x_max"]) / 2 + offset_cm[0]
    cy_world = (b["y_min"] + b["y_max"]) / 2 + offset_cm[1]
    cx, cy = world_to_canvas(cx_world, cy_world)

    dims = fc.UXO_SIZE_TABLE_MM[uxo_type]
    w_px = max(4, int((dims["w"] / 10.0) * PX_PER_CM))
    d_px = max(4, int((dims["d"] / 10.0) * PX_PER_CM))

    if uxo_type == "missile":
        _draw_missile(canvas, cx, cy, w_px, d_px)
    elif uxo_type == "dumb":
        _draw_mortar_round(canvas, cx, cy, w_px, d_px)
    else:  # cluster
        _draw_submunition(canvas, cx, cy, w_px, d_px)

    return cx_world, cy_world


# =====================================================================
# 시설물 아이콘 (관제탑/관제레이더/격납고/일반건물/무기고) + 화재/파손 오버레이
# =====================================================================
def _draw_control_tower(canvas, p1, p2):
    """관제탑: 좁은 기둥 + 상단 관제실 캡"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    cx = (p1[0] + p2[0]) // 2
    cab_h = max(8, h // 4)
    cab_w = max(14, w // 3)
    cab_top = p1[1] + h // 5
    pole_w = max(5, w // 8)

    cv2.rectangle(canvas, (cx - pole_w // 2, cab_top + cab_h), (cx + pole_w // 2, p2[1]), (150, 150, 150), -1)
    cv2.rectangle(canvas, (cx - cab_w // 2, cab_top), (cx + cab_w // 2, cab_top + cab_h), (225, 210, 170), -1)
    cv2.rectangle(canvas, (cx - cab_w // 2, cab_top), (cx + cab_w // 2, cab_top + cab_h), (120, 100, 70), 1)
    cv2.line(canvas, (cx, cab_top), (cx, max(p1[1], cab_top - 10)), (90, 90, 90), 1)


def _draw_radar(canvas, p1, p2):
    """관제레이더: 원형 레이돔 + 받침대"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    base_w = max(6, w // 5)

    cv2.rectangle(canvas, (cx - base_w // 2, cy), (cx + base_w // 2, p2[1]), (150, 150, 150), -1)
    r = max(8, min(w, h) // 3)
    cv2.circle(canvas, (cx, cy), r, (238, 238, 238), -1)
    cv2.circle(canvas, (cx, cy), r, (165, 165, 165), 1)
    cv2.ellipse(canvas, (cx, cy), (r, max(2, r // 3)), 0, 0, 360, (165, 165, 165), 1)


def _draw_hangar(canvas, p1, p2):
    """격납고: 넓은 몸체 + 아치형(반원) 지붕"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    mid_y = p1[1] + h // 2
    cv2.rectangle(canvas, (p1[0], mid_y), p2, (218, 218, 218), -1)

    cx = (p1[0] + p2[0]) // 2
    axes = (w // 2, h // 2)
    cv2.ellipse(canvas, (cx, mid_y), axes, 0, 180, 360, (205, 145, 60), -1)   # 파란(BGR) 아치 지붕
    cv2.ellipse(canvas, (cx, mid_y), axes, 0, 180, 360, (150, 100, 30), 2)
    cv2.rectangle(canvas, (cx - max(4, w // 10), p2[1] - max(4, h // 4)),
                  (cx + max(4, w // 10), p2[1]), (110, 110, 110), -1)  # 격납고 출입구


def _draw_generic_building(canvas, p1, p2, roof_color):
    """일반건물/무기고: 사각 몸체 + 지붕 띠 + 작은 창문"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    cv2.rectangle(canvas, p1, p2, (228, 228, 228), -1)
    roof_h = max(5, h // 4)
    cv2.rectangle(canvas, p1, (p2[0], p1[1] + roof_h), roof_color, -1)

    win_color = (190, 150, 90)
    n_win = 3
    for i in range(n_win):
        wx = p1[0] + w * (i + 1) // (n_win + 1) - 3
        wy = p1[1] + roof_h + h // 4
        cv2.rectangle(canvas, (wx, wy), (wx + 6, wy + 6), win_color, -1)


def _draw_fire_overlay(canvas, p1, p2):
    """화재 상태: 건물 규모에 맞는 크기의 불꽃 아이콘을 건물 위에 겹쳐 그림
    (FA 구역 폭이 건물 자체보다 훨씬 넓을 수 있어 min(w,h) 기준으로 크기를 제한)"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    flame_w = min(w, h) * 1.1
    flame_h = h * 0.8
    cx = (p1[0] + p2[0]) // 2
    base_y = int(p1[1] + h * 0.55)
    top_y = int(base_y - flame_h)

    outer = np.array([
        [cx, top_y],
        [cx + int(flame_w * 0.32), int(base_y - flame_h * 0.28)],
        [cx + int(flame_w * 0.18), base_y],
        [cx - int(flame_w * 0.18), base_y],
        [cx - int(flame_w * 0.32), int(base_y - flame_h * 0.28)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [outer], (0, 85, 230))  # 주황/빨강 (BGR)

    inner = np.array([
        [cx, int(top_y + flame_h * 0.25)],
        [cx + int(flame_w * 0.14), int(base_y - flame_h * 0.32)],
        [cx - int(flame_w * 0.14), int(base_y - flame_h * 0.32)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [inner], (0, 210, 255))  # 노란 불꽃 심지


def _draw_damage_overlay(canvas, p1, p2):
    """파손 상태: 지붕/구조물 붕괴 및 잔해 표현 (하단 영역을 어둡게)"""
    w, h = p2[0] - p1[0], p2[1] - p1[1]
    collapse_top = p1[1] + int(h * 0.35)
    cv2.rectangle(canvas, (p1[0], collapse_top), (p2[0], p2[1]), (28, 28, 28), -1)

    rng = _seeded_rng(p1[0], p1[1], p2[0], p2[1])
    for _ in range(6):
        cx = int(rng.integers(p1[0], p2[0]))
        cy = int(rng.integers(collapse_top, p2[1]))
        r = max(2, int(min(w, h) * 0.08))
        cv2.circle(canvas, (cx, cy), r, (12, 12, 12), -1)


def draw_facility_state(canvas, slot, state="normal"):
    """FA 슬롯에 시설물 종류별 아이콘을 그리고, 상태에 따라 화재/파손 오버레이를 얹음"""
    b = fc.SEGMENTS[slot]
    p1 = world_to_canvas(b["x_min"] + 5, b["y_min"] + 5)
    p2 = world_to_canvas(b["x_max"] - 5, b["y_max"] - 5)
    facility_type = fc.FACILITY_TYPE_BY_SLOT[slot]

    if facility_type == "control_tower":
        _draw_control_tower(canvas, p1, p2)
    elif facility_type == "radar":
        _draw_radar(canvas, p1, p2)
    elif facility_type == "hangar":
        _draw_hangar(canvas, p1, p2)
    elif facility_type == "weapon_depot":
        _draw_generic_building(canvas, p1, p2, roof_color=(115, 115, 130))
    else:
        _draw_generic_building(canvas, p1, p2, roof_color=(170, 170, 170))

    if state == "fire":
        _draw_fire_overlay(canvas, p1, p2)
    elif state == "destroy":
        _draw_damage_overlay(canvas, p1, p2)

    cv2.putText(canvas, f"{slot}:{state}", (p1[0], p2[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def generate_scene(seed=0, fire_flicker_phase=0.0):
    dict_id = getattr(cv2.aruco, fc.ARUCO_DICT_NAME)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    canvas_w = fc.FIELD_WIDTH_CM * PX_PER_CM + 2 * MARGIN_PX
    canvas_h = fc.FIELD_HEIGHT_CM * PX_PER_CM + 2 * MARGIN_PX
    canvas = np.full((canvas_h, canvas_w, 3), (110, 150, 95), dtype=np.uint8)  # 잔디 여백

    draw_field_base(canvas)

    # ---- 시나리오 배치 ----
    draw_crater(canvas, "RW-03", "medium")
    draw_crater(canvas, "RW-07", "big")

    draw_uxo(canvas, "RW-05", "cluster")
    draw_uxo(canvas, "TW-B2", "dumb")
    draw_uxo(canvas, "RW-09", "missile", offset_cm=(15, 0))

    draw_facility_state(canvas, "FA-01", "normal")
    # 화재는 깜빡임을 흉내내기 위해 프레임마다 밝기를 살짝 변화
    fire_state = "fire"
    draw_facility_state(canvas, "FA-02", fire_state)
    draw_facility_state(canvas, "FA-03", "destroy")
    draw_facility_state(canvas, "FA-04", "normal")
    draw_facility_state(canvas, "FA-05", "normal")
    draw_facility_state(canvas, "FA-06", "normal")

    draw_aruco_markers(canvas, aruco_dict)

    # 프레임 간 미세한 노이즈(카메라 센서 노이즈 흉내) 추가 - 재현성 위해 seed 사용
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.5, canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 화재 영역만 프레임마다 밝기를 흔들어 '깜빡임'을 시뮬레이션
    b = fc.SEGMENTS["FA-02"]
    p1 = world_to_canvas(b["x_min"] + 5, b["y_min"] + 5)
    p2 = world_to_canvas(b["x_max"] - 5, b["y_max"] - 5)
    flicker = int(30 * np.sin(fire_flicker_phase))
    noisy[p1[1]:p2[1], p1[0]:p2[0]] = np.clip(
        noisy[p1[1]:p2[1], p1[0]:p2[0]].astype(np.int16) + flicker, 0, 255
    ).astype(np.uint8)

    return noisy


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(4):
        frame = generate_scene(seed=i, fire_flicker_phase=i * 1.6)
        path = os.path.join(out_dir, f"frame_{i:02d}.png")
        imwrite_safe(path, frame)
        frames.append(path)
        print(f"생성됨: {path}  (shape={frame.shape})")
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    main(out)
