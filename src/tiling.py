# -*- coding: utf-8 -*-
"""
tiling.py
=========
탑뷰(bird's eye) 워핑 이미지에서 zone 타일(3-A)과 시설물 ROI(3-B)를 잘라냅니다.

워핑 단계(calibration.warp_to_topview)를 거치고 나면 탑뷰 이미지의 픽셀좌표가
경기장 실좌표(cm) * px_per_cm 와 정확히 같으므로, 여기서는 프레임별 호모그래피
역투영 없이 단순 픽셀 슬라이싱만으로 crop할 수 있습니다.
zone 경계는 서로 걸치지 않는 단순 그리드이므로 overlap 처리가 필요 없습니다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


def _world_bbox_to_topview_px(bounds: dict, px_per_cm: float, canvas_w: int, canvas_h: int):
    x0 = int(round(bounds["x_min"] * px_per_cm))
    y0 = int(round(bounds["y_min"] * px_per_cm))
    x1 = int(round(bounds["x_max"] * px_per_cm))
    y1 = int(round(bounds["y_max"] * px_per_cm))
    x0 = max(0, min(x0, canvas_w))
    x1 = max(0, min(x1, canvas_w))
    y0 = max(0, min(y0, canvas_h))
    y1 = max(0, min(y1, canvas_h))
    return x0, y0, x1, y1


def crop_zone_tiles(topview_bgr, px_per_cm: float, zone_order: list = None) -> dict:
    """
    3-A: 활주로/유도로 zone 타일 crop (경계 걸침 없는 단순 그리드 분할).
    반환: {zone_name: tile_image_bgr, ...}  -- 순서는 zone_order(기본 fc.ZONE_TILE_ORDER)를 따름
    """
    zone_order = zone_order or fc.ZONE_TILE_ORDER
    h, w = topview_bgr.shape[:2]
    tiles = {}
    for zone_name in zone_order:
        bounds = fc.SEGMENTS[zone_name]
        x0, y0, x1, y1 = _world_bbox_to_topview_px(bounds, px_per_cm, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        tiles[zone_name] = topview_bgr[y0:y1, x0:x1]
    return tiles


def crop_facility_rois(topview_bgr, px_per_cm: float, slots: list = None) -> dict:
    """
    3-B: 시설물 고정 좌표(FA-01~06) crop.
    반환: {slot: roi_image_bgr, ...}
    """
    slots = slots or fc.FACILITY_SLOTS
    h, w = topview_bgr.shape[:2]
    rois = {}
    for slot in slots:
        bounds = fc.SEGMENTS[slot]
        x0, y0, x1, y1 = _world_bbox_to_topview_px(bounds, px_per_cm, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        rois[slot] = topview_bgr[y0:y1, x0:x1]
    return rois


def tile_origin_world_cm(zone_name: str):
    """타일 내부 로컬 픽셀좌표를 전역 실좌표로 환산할 때 더해줄 zone의 좌상단 실좌표(cm)"""
    b = fc.SEGMENTS[zone_name]
    return b["x_min"], b["y_min"]
