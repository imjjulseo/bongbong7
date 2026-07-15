# -*- coding: utf-8 -*-
"""
calibration.py
==============
ArUco 마커로 카메라 픽셀 좌표 <-> 경기장 실좌표(cm)를 변환하는 호모그래피를 계산합니다.
드론이 프레임마다 미세하게 움직일 수 있어, calibrate_from_image()로 매 프레임 재보정합니다.
"""
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


def _get_aruco_dictionary():
    dict_id = getattr(cv2.aruco, fc.ARUCO_DICT_NAME)
    return cv2.aruco.getPredefinedDictionary(dict_id)


class FieldCalibrator:
    def __init__(self):
        self.aruco_dict = _get_aruco_dictionary()
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.homography = None  # 3x3, pixel(x,y,1) -> world_cm(x,y,1)
        self.last_marker_corners = []  # 최근 검출된 마커들의 픽셀 코너 (마스킹용)

    # -----------------------------------------------------------------
    def detect_markers(self, image_bgr: np.ndarray):
        """이미지에서 ArUco 마커를 검출. corners, ids 반환."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        return corners, ids

    # -----------------------------------------------------------------
    def calibrate_from_image(self, image_bgr: np.ndarray, refine_subpixel: bool = True):
        """
        이미지 속 마커 중심좌표(픽셀) <-> 알려진 실좌표(cm)로 호모그래피를 계산.
        최소 4개 마커(경기장 모서리)가 필요.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(image_bgr)

        if ids is None or len(ids) < 4:
            raise RuntimeError(
                f"ArUco 마커가 4개 미만 검출됨(검출:{0 if ids is None else len(ids)}). "
                "드론 고도/각도를 조정하거나 조명을 확인하세요."
            )

        # 서브픽셀 코너 정밀화 (호모그래피 오차를 줄이는 핵심 포인트)
        if refine_subpixel:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            for c in corners:
                cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria)
        
        pixel_pts = []
        world_pts = []
        marker_corner_list = []
        
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            marker_corner_list.append(marker_corners[0]) 
            
            if marker_id not in fc.ARUCO_MARKER_WORLD_POSITIONS:
                continue 
            
            # [수정된 부분] 중심점 대신 마커의 특정 코너를 가져옴
            # config에 정의된 인덱스를 사용하되, 없으면 기본값 0(좌상단) 사용
            corner_idx = fc.ARUCO_MARKER_CORNER_INDEX.get(marker_id, 0)
            target_pixel = marker_corners[0][corner_idx] # (x, y) 픽셀좌표
            
            pixel_pts.append(target_pixel)
            world_pts.append(fc.ARUCO_MARKER_WORLD_POSITIONS[marker_id])
            
        self.last_marker_corners = marker_corner_list

        if len(pixel_pts) < 4:
            raise RuntimeError("설정된 마커 ID와 일치하는 마커가 4개 미만입니다.")

        pixel_pts = np.array(pixel_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        H, mask = cv2.findHomography(pixel_pts, world_pts, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("호모그래피 계산 실패 (마커 배치를 확인하세요)")

        self.homography = H
        return H

    # -----------------------------------------------------------------
    def pixel_to_world(self, px: float, py: float):
        """픽셀좌표 -> 경기장 실좌표(cm)"""
        if self.homography is None:
            raise RuntimeError("먼저 calibrate_from_image()를 호출하세요.")
        pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pt, self.homography)
        return float(world[0, 0, 0]), float(world[0, 0, 1])

    def world_to_pixel(self, wx: float, wy: float):
        """실좌표(cm) -> 픽셀좌표. 위치가 고정된 시설물의 ROI를 역으로 찾을 때 사용."""
        if self.homography is None:
            raise RuntimeError("먼저 calibrate_from_image()를 호출하세요.")
        H_inv = np.linalg.inv(self.homography)
        pt = np.array([[wx, wy]], dtype=np.float32).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(pt, H_inv)
        return float(px[0, 0, 0]), float(px[0, 0, 1])

    def world_bbox_to_pixel_bbox(self, x_min, y_min, x_max, y_max, image_shape):
        """실좌표 사각형(FA 슬롯 등) -> 픽셀 bbox(x,y,w,h), 이미지 경계 밖으로 나가지 않도록 클램프"""
        corners_world = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        corners_px = np.array([self.world_to_pixel(wx, wy) for wx, wy in corners_world])
        h, w = image_shape[:2]
        px_x_min = int(np.clip(corners_px[:, 0].min(), 0, w - 1))
        px_x_max = int(np.clip(corners_px[:, 0].max(), 0, w - 1))
        px_y_min = int(np.clip(corners_px[:, 1].min(), 0, h - 1))
        px_y_max = int(np.clip(corners_px[:, 1].max(), 0, h - 1))
        return px_x_min, px_y_min, max(1, px_x_max - px_x_min), max(1, px_y_max - px_y_min)

    # -----------------------------------------------------------------
    def warp_to_topview(self, image_bgr: np.ndarray, px_per_cm: float = None):
        """
        2단계 핵심: 원본 프레임을 '경기장 실좌표계와 픽셀이 1:1로 정렬된' 탑뷰(bird's eye)
        이미지로 워핑합니다. 이 결과물의 픽셀좌표는 (world_cm * px_per_cm)와 정확히 같으므로,
        이후 zone 타일 crop(3-A)이나 시설물 ROI crop(3-B)은 프레임별 역투영 없이
        단순 픽셀 슬라이싱만으로 계산할 수 있습니다.

        pixel_to_world가 만드는 호모그래피 H(pixel->world_cm) 앞에 스케일 행렬을 곱해
        pixel->topview_pixel 호모그래피를 만들고, 그걸로 이미지 자체를 워핑합니다.

        반환: (topview_bgr, px_per_cm)
        """
        if self.homography is None:
            raise RuntimeError("먼저 calibrate_from_image()를 호출하세요.")
        px_per_cm = px_per_cm if px_per_cm is not None else fc.WARP_PX_PER_CM
        canvas_w = int(round(fc.FIELD_WIDTH_CM * px_per_cm))
        canvas_h = int(round(fc.FIELD_HEIGHT_CM * px_per_cm))

        scale = np.array([
            [px_per_cm, 0, 0],
            [0, px_per_cm, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        H_topview = scale @ self.homography

        topview = cv2.warpPerspective(image_bgr, H_topview, (canvas_w, canvas_h))
        return topview, px_per_cm


def estimate_camera_angle(pipeline):
    """
    파이프라인의 Calibrator에 저장된 마지막 ArUco 마커 코너 비율을 통해
    카메라가 90도(TOP)인지 45도(SIDE)인지 판별합니다.
    """
    try:
        corners = pipeline.calibrator.last_marker_corners
        if corners is None or len(corners) == 0:
            return "TOP_VIEW"

        angles = []
        for corner_set in corners:
            c = corner_set.reshape(4, 2)
            top_edge = np.linalg.norm(c[0] - c[1])
            bottom_edge = np.linalg.norm(c[3] - c[2])
            left_edge = np.linalg.norm(c[0] - c[3])
            right_edge = np.linalg.norm(c[1] - c[2])

            avg_w = (top_edge + bottom_edge) / 2.0
            avg_h = (left_edge + right_edge) / 2.0
            ratio = avg_h / avg_w if avg_w > 0 else 0
            # 겉보기 비율이 1에 가까우면 수직, 압축되어 0.85 미만이면 측면으로 간주
            if ratio > 0.85:
                angles.append("TOP_VIEW")
            else:
                angles.append("SIDE_VIEW")

        # 다수결 판별
        if angles.count("TOP_VIEW") > len(angles) / 2:
            return "TOP_VIEW"
        else:
            return "SIDE_VIEW"
    except Exception as e:
        print(f"[video_watcher] 각도 판별 오류 (기본값 TOP_VIEW 사용): {e}")
        return "TOP_VIEW"
