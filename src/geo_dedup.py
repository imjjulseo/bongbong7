# -*- coding: utf-8 -*-
"""
geo_dedup.py
============
드론이 같은 지점을 여러 프레임에 걸쳐 촬영하면 같은 폭파구/불발탄이
여러 번 탐지될 수 있습니다. 픽셀 좌표가 아니라 '실좌표(cm)'로 변환한 뒤
가까운 탐지끼리 하나로 묶어 중복을 제거합니다. (Union-Find 기반 클러스터링)
"""
import numpy as np


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
