# -*- coding: utf-8 -*-
"""
geo_dedup.py
============
드론이 같은 지점을 여러 프레임에 걸쳐 촬영하면 같은 폭파구/불발탄이
여러 번 탐지될 수 있습니다. 픽셀 좌표가 아니라 '실좌표(cm)'로 변환한 뒤
가까운 탐지끼리 하나로 묶어 중복을 제거합니다. (Union-Find 기반 클러스터링)
"""
import numpy as np
from collections import defaultdict


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def dedup_by_world_distance(detections: list, distance_threshold_cm: float = 5.0):
    """
    detections: [{"world_xy": (x_cm, y_cm), ...다른 필드...}, ...]
    distance_threshold_cm보다 가까운 탐지들은 같은 물체로 간주해 하나로 병합합니다.
    병합 시 신뢰도(confidence)가 가장 높은 탐지의 정보를 대표값으로 사용합니다.

    반환: 중복 제거된 detections 리스트 (각 항목에 "merged_count" 필드 추가)
    """
    n = len(detections)
    if n == 0:
        return []

    coords = np.array([d["world_xy"] for d in detections], dtype=np.float64)
    uf = UnionFind(n)

    # 단순 O(n^2) 거리 비교 - 한 임무당 탐지 개수가 많지 않아 충분히 빠름
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= distance_threshold_cm:
                uf.union(i, j)

    clusters = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    merged = []
    for idx_list in clusters.values():
        group = [detections[i] for i in idx_list]
        # 신뢰도가 있으면 최고 신뢰도, 없으면 그냥 첫 번째를 대표로 사용
        group_sorted = sorted(group, key=lambda d: d.get("confidence", 1.0), reverse=True)
        representative = dict(group_sorted[0])
        # 대표 좌표는 클러스터 평균으로 스무딩(더 안정적인 위치 추정)
        avg_xy = coords[idx_list].mean(axis=0)
        representative["world_xy"] = (float(avg_xy[0]), float(avg_xy[1]))
        representative["merged_count"] = len(group)
        merged.append(representative)

    return merged


def dedup_by_zone(detections: list, class_key: str):
    """
    각 구역(segment)당 단 하나의 객체만 남기도록 프레임 간 앙상블을 수행합니다.
    class_key: 폭파구의 경우 'size_class', 불발탄의 경우 'type'
    """
    if not detections:
        return []

    # 1. 구역(segment)별로 탐지된 결과를 그룹화
    zone_groups = defaultdict(list)
    for d in detections:
        zone = d.get("segment")
        if zone:
            zone_groups[zone].append(d)

    merged = []
    for zone, group in zone_groups.items():
        # 2. 세부 클래스별 누적 신뢰도(Confidence Sum) 계산
        score_sums = defaultdict(float)
        for d in group:
            cls_val = d.get(class_key)
            conf = d.get("confidence", 1.0)
            score_sums[cls_val] += conf
        
        # 3. 누적 신뢰도가 가장 높은 세부 클래스 최종 선정
        best_class = max(score_sums.items(), key=lambda x: x[1])[0]
        
        # 4. 선정된 클래스 그룹 내에서 대표값 추출 및 좌표 스무딩
        best_class_detections = [d for d in group if d.get(class_key) == best_class]
        best_class_detections.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        
        representative = dict(best_class_detections[0]) # 최고 신뢰도 객체를 베이스로 사용
        
        # 대표 실좌표는 해당 클래스로 판별된 프레임들의 평균 좌표 사용
        avg_x = sum(d["world_xy"][0] for d in best_class_detections) / len(best_class_detections)
        avg_y = sum(d["world_xy"][1] for d in best_class_detections) / len(best_class_detections)
        
        representative["world_xy"] = (float(avg_x), float(avg_y))
        representative["merged_count"] = len(group)
        representative["accumulated_confidence"] = score_sums[best_class]
        
        merged.append(representative)

    return merged