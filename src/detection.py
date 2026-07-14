# -*- coding: utf-8 -*-
"""
detection.py
============
객체 탐지/분류 모듈입니다. "고전 CV"와 "YOLO" 두 백엔드를 같은 인터페이스로 감싸서,
대회 현장에서 테스트 기간 중 촬영한 이미지로 YOLO 학습이 끝나면 config 값 하나만 바꿔
파이프라인 코드 수정 없이 전환할 수 있도록 설계했습니다.

  - 폭파구/불발탄 통합 탐지 : ObjectDetectorBackend
      · ClassicalBlobDetector : 색상/형태 기반 탐지 (지금 바로 동작, 학습 불필요)
      · YoloObjectDetector    : ultralytics YOLO 가중치가 준비되면 사용
  - 시설물 상태 분류 : FacilityStatusBackend
      · ClassicalFacilityClassifier : HSV 색상 + 시계열 깜빡임 검사
      · YoloFacilityClassifier      : ultralytics YOLO(분류/탐지) 가중치가 준비되면 사용

전환 방법: config/field_config.py의 DETECTOR_BACKEND / FACILITY_BACKEND를
"classical" -> "yolo"로 바꾸면 build_object_detector()/build_facility_classifier()가
자동으로 다른 구현체를 반환합니다. ultralytics는 "yolo" 백엔드를 실제로 쓸 때만
import하므로, 고전 CV만 쓰는 동안은 설치되어 있지 않아도 됩니다.

왜 고전 CV를 기본값으로 했는가:
  - 대회 PC에 GPU/클라우드 사용이 금지되어 있고, 대회 전 실측 데이터가 제한적이라
    딥러닝 모델의 신뢰도가 흔들릴 수 있음
  - 폭파구(검정 불규칙 blob), 불발탄(뚜렷한 기하학적 형태), 화재(밝은 적/주황색)는
    색상+형태만으로도 상당히 안정적으로 탐지 가능
  - "AI 예측이 실패해도 동작하는 폴백"이라는 안전장치 역할도 겸함
"""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


# =====================================================================
# 공통 유틸: 형태 기술자(shape descriptor)
# =====================================================================
def shape_descriptors(contour):
    """윤곽선의 원형도(circularity), 종횡비(aspect ratio), 등가지름을 계산"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)  # 1.0에 가까울수록 원에 가까움

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(1e-6, min(w, h))
    equiv_diameter = np.sqrt(4 * area / np.pi)
    return {
        "area_px": area,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "equiv_diameter_px": equiv_diameter,
        "long_axis_px": max(w, h),
        "bbox": (x, y, w, h),
    }


def detect_dark_blobs(image_bgr: np.ndarray, dark_threshold=80, min_area_px=40, max_area_px=20000):
    """폭파구/불발탄 모두 어두운 물체라는 공통점을 이용한 통합 blob 탐지 (고전 CV 백엔드용)"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        desc = shape_descriptors(c)
        if not (min_area_px < desc["area_px"] < max_area_px):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        blobs.append({
            "center_px": (cx, cy),
            "equiv_diameter_px": desc["equiv_diameter_px"],
            "long_axis_px": desc["long_axis_px"],
            "aspect_ratio": desc["aspect_ratio"],
            "circularity": desc["circularity"],
            "contour": c,
        })
    return blobs


def mask_out_regions(image_bgr: np.ndarray, polygons: list, fill_value=255):
    """
    ArUco 마커처럼 '물체가 아닌데 어둡게 보이는' 영역을 미리 지워서(흰색 등으로 채움)
    오탐을 방지합니다. polygons: [Nx2 array, ...] 픽셀좌표 다각형 리스트 (여유있게 살짝 확대해서 사용 권장)
    """
    if not polygons:
        return image_bgr
    masked = image_bgr.copy()
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masked, [pts], (fill_value, fill_value, fill_value))
    return masked


def classify_blob(diameter_mm: float, long_axis_mm: float, aspect_ratio: float):
    """
    실측 mm 크기 + 종횡비를 기준으로 '폭파구' 또는 '불발탄' 중 어느 쪽에 더 가까운지,
    그리고 세부 종류(대형/중형/소형 또는 자탄/포탄/미사일)를 함께 판정합니다. (고전 CV 백엔드 전용)

    - 폭파구는 비교적 둥글둥글하므로 등가지름(diameter_mm, 원 기준 환산값)과 비교
    - 불발탄(특히 포탄/미사일)은 길쭉하므로 장축길이(long_axis_mm)와 비교하는 것이 더 정확함
    두 치수표를 동시에 놓고 '가장 가까운 후보'를 전역적으로 찾는 방식이라
    폭파구/불발탄이 서로 중복 집계되지 않습니다.
    """
    candidates = []

    for name, dims in fc.CRATER_SIZE_TABLE_MM.items():
        ref_diameter = (dims["w"] + dims["h"]) / 2.0
        ref_aspect = max(dims["w"], dims["h"]) / min(dims["w"], dims["h"])
        diam_diff = abs(diameter_mm - ref_diameter) / ref_diameter
        ar_diff = abs(aspect_ratio - ref_aspect) / ref_aspect
        score = diam_diff * 0.7 + ar_diff * 0.3   # 폭파구는 크기가 더 결정적
        candidates.append(("crater", name, score))

    for name, dims in fc.UXO_SIZE_TABLE_MM.items():
        ref_length = max(dims["w"], dims["d"])
        # aspect_ratio는 항상 max(w,h)/min(w,h) >= 1로 계산되므로 기준값도 동일한 형태로 맞춤
        ref_aspect = max(dims["w"], dims["d"]) / min(dims["w"], dims["d"])
        len_diff = abs(long_axis_mm - ref_length) / ref_length
        ar_diff = abs(aspect_ratio - ref_aspect) / max(ref_aspect, 0.01)
        score = len_diff * 0.4 + ar_diff * 0.6   # 불발탄은 형태(길쭉함)가 더 결정적
        candidates.append(("uxo", name, score))

    candidates.sort(key=lambda x: x[2])
    category, subtype, score = candidates[0]
    confidence = max(0.3, round(1.0 - min(score, 0.7), 2))
    return category, subtype, confidence


# =====================================================================
# 1. 폭파구/불발탄 통합 탐지 백엔드
# =====================================================================
@dataclass
class ObjectDetection:
    """백엔드에 상관없이 pipeline.py가 다루는 공통 탐지 결과 형태.

    center_px / equiv_diameter_px / long_axis_px : 실좌표(mm, m) 환산에 필요한 픽셀 값
        (호모그래피는 백엔드가 아닌 pipeline이 알고 있으므로 항상 픽셀 단위로 반환)
    category / subtype / confidence : 분류 결과.
        고전 CV 백엔드는 픽셀->mm 변환 이후에야 classify_blob()으로 분류할 수 있어
        이 필드들을 None으로 남겨두고, pipeline이 후처리로 채웁니다.
        YOLO 백엔드는 모델이 이미지 자체에서 바로 클래스를 예측하므로 이 자리에서 채웁니다.
    """
    center_px: Tuple[float, float]
    equiv_diameter_px: float
    long_axis_px: float
    aspect_ratio: float
    category: Optional[str] = None
    subtype: Optional[str] = None
    confidence: Optional[float] = None


class ObjectDetectorBackend(ABC):
    """폭파구/불발탄 통합 탐지기의 공통 인터페이스"""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[ObjectDetection]:
        ...


class ClassicalBlobDetector(ObjectDetectorBackend):
    """색상/형태 기반 탐지 (지금 바로 동작, 학습 불필요).
    분류(category/subtype/confidence)는 실좌표 mm 변환이 필요해 pipeline에서 classify_blob()으로 채움."""

    def __init__(self, dark_threshold=80, min_area_px=40, max_area_px=20000):
        self.dark_threshold = dark_threshold
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

    def detect(self, image_bgr: np.ndarray) -> List[ObjectDetection]:
        blobs = detect_dark_blobs(image_bgr, self.dark_threshold, self.min_area_px, self.max_area_px)
        return [
            ObjectDetection(
                center_px=b["center_px"],
                equiv_diameter_px=b["equiv_diameter_px"],
                long_axis_px=b["long_axis_px"],
                aspect_ratio=b["aspect_ratio"],
            )
            for b in blobs
        ]


class YoloObjectDetector(ObjectDetectorBackend):
    """
    ultralytics YOLO 가중치가 준비되면 사용하는 백엔드.
    학습 클래스는 fc.YOLO_OBJECT_CLASS_MAP(class idx -> (category, subtype))에 맞춰 정의하고,
    data.yaml의 names 순서와 반드시 일치시켜야 합니다.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = None,
                 class_map: dict = None):
        from ultralytics import YOLO  # "yolo" 백엔드를 실제로 쓸 때만 의존성 필요

        self.weights_path = weights_path or fc.YOLO_OBJECT_WEIGHTS
        self.conf_threshold = conf_threshold if conf_threshold is not None else fc.YOLO_OBJECT_CONF_THRESHOLD
        self.class_map = class_map or fc.YOLO_OBJECT_CLASS_MAP
        self.model = YOLO(self.weights_path)

    def detect(self, image_bgr: np.ndarray) -> List[ObjectDetection]:
        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cx, cy, w, h = [float(v) for v in box.xywh[0].tolist()]
            cls_idx = int(box.cls[0])
            if cls_idx not in self.class_map:
                continue  # 학습 클래스 매핑에 없는 인덱스는 무시
            category, subtype = self.class_map[cls_idx]
            out.append(ObjectDetection(
                center_px=(cx, cy),
                equiv_diameter_px=float(np.sqrt(max(w, 1e-6) * max(h, 1e-6))),
                long_axis_px=max(w, h),
                aspect_ratio=max(w, h) / max(1e-6, min(w, h)),
                category=category,
                subtype=subtype,
                confidence=round(float(box.conf[0]), 2),
            ))
        return out


def build_object_detector(backend: str = None, **kwargs) -> ObjectDetectorBackend:
    """config.DETECTOR_BACKEND(또는 인자로 넘긴 backend)에 맞는 탐지기를 생성"""
    backend = backend or fc.DETECTOR_BACKEND
    if backend == "classical":
        return ClassicalBlobDetector(**kwargs)
    if backend == "yolo":
        return YoloObjectDetector(**kwargs)
    raise ValueError(f"알 수 없는 DETECTOR_BACKEND: {backend!r} (classical|yolo 중 하나여야 함)")


# =====================================================================
# 2. 시설물 상태 분류 백엔드 (정상/파손/화재)
# =====================================================================
class FacilityStatusBackend(ABC):
    """시설물 상태 분류기의 공통 인터페이스"""

    @abstractmethod
    def classify(self, rois_bgr: List[np.ndarray]) -> Tuple[str, float]:
        """rois_bgr: 같은 시설물을 여러 프레임에서 각각 잘라낸(crop) ROI 이미지 리스트.
        반환: (status, confidence). status는 fc.FACILITY_STATUS_OPTIONS 중 하나(미확인 제외)."""
        ...


class ClassicalFacilityClassifier(FacilityStatusBackend):
    """HSV 색상 기반 1차 판정 + 시계열 화재 깜빡임 검사 (지금 바로 동작, 학습 불필요)"""

    def classify_single_frame(self, roi_bgr: np.ndarray):
        """
        roi_bgr: 이미 잘라낸(crop된) 시설물 영역 이미지.
        프레임마다 호모그래피가 다를 수 있으므로, ROI 자르기는 pipeline 쪽에서
        프레임별로 수행하고 이 함수에는 잘라진 결과만 넘깁니다.
        화재(밝은 적/주황), 파손(불규칙 어두운 패치 비율), 정상 중 하나를 1차 판정.
        """
        roi = roi_bgr
        if roi.size == 0:
            return "미확인", 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 화재색 범위 (빨강~주황, 높은 채도/명도)
        fire_mask1 = cv2.inRange(hsv, (0, 120, 150), (15, 255, 255))
        fire_mask2 = cv2.inRange(hsv, (160, 120, 150), (179, 255, 255))
        fire_ratio = (cv2.countNonZero(fire_mask1) + cv2.countNonZero(fire_mask2)) / max(1, roi.shape[0] * roi.shape[1])

        if fire_ratio > 0.03:
            return "화재", round(min(1.0, fire_ratio * 10), 2)

        # 파손 추정: 어두운 불규칙 영역 비율(그림자/균열/붕괴 등으로 어두워짐)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.mean(gray < 60)
        if dark_ratio > 0.35:
            return "파손", round(min(1.0, dark_ratio), 2)

        return "정상", round(1.0 - dark_ratio, 2)

    def classify(self, rois_bgr: list):
        """
        여러 프레임에 걸쳐 '화재 깜빡임'을 확인하여 오탐(빨간 벽돌 등)을 걸러냄.
        rois_bgr: 프레임별로 각각 잘라낸(이미 crop된) 같은 시설물의 ROI 이미지 리스트
        """
        results = [self.classify_single_frame(r) for r in rois_bgr]
        labels = [r[0] for r in results]
        confidences = [r[1] for r in results]

        if labels.count("화재") == 0:
            # 화재가 아니면 다수결로 결정
            most_common = Counter(labels).most_common(1)[0][0]
            avg_conf = float(np.mean(confidences))
            return most_common, round(avg_conf, 2)

        # 화재로 판정된 프레임 비율이 충분히 높고, 밝기 변화(깜빡임)가 감지되면 화재 확정
        fire_ratio_frames = labels.count("화재") / len(labels)
        brightness_series = [np.mean(r) for r in rois_bgr]
        flicker = float(np.std(brightness_series))

        if fire_ratio_frames > 0.3 and flicker > 1.0:
            return "화재", round(float(np.mean(confidences)), 2)
        else:
            # 화재로 보였지만 깜빡임이 없다면 정지된 붉은 물체(오탐 가능성) -> 파손/정상 재판정
            non_fire_labels = [l for l in labels if l != "화재"]
            if non_fire_labels:
                return Counter(non_fire_labels).most_common(1)[0][0], 0.5
            return "파손", 0.4


class YoloFacilityClassifier(FacilityStatusBackend):
    """
    ultralytics YOLO(분류 모델 권장: yolov8n-cls 등) 가중치가 준비되면 사용하는 백엔드.
    프레임별로 예측한 뒤 다수결로 최종 상태를 정하고, 다수결에 참여한 프레임들의
    평균 신뢰도를 반환합니다. 학습 클래스는 fc.YOLO_FACILITY_CLASS_MAP에 맞춰 정의합니다.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = None,
                 class_map: dict = None):
        from ultralytics import YOLO  # "yolo" 백엔드를 실제로 쓸 때만 의존성 필요

        self.weights_path = weights_path or fc.YOLO_FACILITY_WEIGHTS
        self.conf_threshold = conf_threshold if conf_threshold is not None else fc.YOLO_FACILITY_CONF_THRESHOLD
        self.class_map = class_map or fc.YOLO_FACILITY_CLASS_MAP
        self.model = YOLO(self.weights_path)

    def _classify_single_frame(self, roi_bgr: np.ndarray):
        if roi_bgr.size == 0:
            return "미확인", 0.0
        result = self.model.predict(roi_bgr, conf=self.conf_threshold, verbose=False)[0]
        if getattr(result, "probs", None) is not None:  # 분류(cls) 모델
            cls_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
        elif len(result.boxes) > 0:  # 탐지(detect) 모델로 대체한 경우 - 가장 신뢰도 높은 box 사용
            best = max(result.boxes, key=lambda b: float(b.conf[0]))
            cls_idx = int(best.cls[0])
            confidence = float(best.conf[0])
        else:
            return "미확인", 0.0
        status = self.class_map.get(cls_idx, "미확인")
        return status, round(confidence, 2)

    def classify(self, rois_bgr: List[np.ndarray]) -> Tuple[str, float]:
        results = [self._classify_single_frame(r) for r in rois_bgr]
        labels = [r[0] for r in results]
        confidences = [r[1] for r in results]
        most_common = Counter(labels).most_common(1)[0][0]
        matching_conf = [c for l, c in zip(labels, confidences) if l == most_common]
        return most_common, round(float(np.mean(matching_conf)), 2)


def build_facility_classifier(backend: str = None, **kwargs) -> FacilityStatusBackend:
    """config.FACILITY_BACKEND(또는 인자로 넘긴 backend)에 맞는 분류기를 생성"""
    backend = backend or fc.FACILITY_BACKEND
    if backend == "classical":
        return ClassicalFacilityClassifier(**kwargs)
    if backend == "yolo":
        return YoloFacilityClassifier(**kwargs)
    raise ValueError(f"알 수 없는 FACILITY_BACKEND: {backend!r} (classical|yolo 중 하나여야 함)")
