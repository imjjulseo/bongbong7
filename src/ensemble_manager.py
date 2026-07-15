import copy
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc

class VideoEnsembleManager:
    """최근 N개의 영상 파이프라인 결과를 유지하며 각도(Angle)와 신뢰도(Confidence) 기반으로 결과를 융합합니다."""
    def __init__(self, max_history=2, conf_threshold=fc.CONFIDENCE_THRESHOLD):
        self.history = deque(maxlen=max_history)
        # 측면 뷰(Side View) 판독이 필수적인 시설물 목록 (테스트 결과 반영)
        self.TOP_DEPENDENT_FACILITIES = ["FA-03"] 
        self.SIDE_DEPENDENT_FACILITIES = ["FA-01"] 
        self.conf_threshold = conf_threshold

    def add_video_result(self, json_outputs, angle_type, confidence):
        """새로운 영상의 JSON 결과와 판별된 각도를 히스토리에 추가"""
        # print("Confidence added: ")
        # print(confidence)
        self.history.append({
            "data": json_outputs,
            "angle": angle_type, 
            "confidence" : confidence, 
        })
    def get_ensembled_result(self):
        """히스토리에 쌓인 결과를 바탕으로 최적의 병합된 JSON 생성"""
        if not self.history:
            return None
            
        # 가장 최근 결과를 베이스로 깊은 복사
        latest_result = copy.deepcopy(self.history[-1]["data"])
        latest_angle = self.history[-1]["angle"]
        latest_conf = self.history[-1]["confidence"]
        
        # 히스토리가 1개뿐이면 앙상블 없이 그대로 반환
        if len(self.history) == 1:
            return latest_result
        
        prev_result = self.history[0]["data"]
        prev_angle = self.history[0]["angle"]
        prev_conf = self.history[0]["confidence"]

        # [앙상블 룰 1] 측면 뷰 필수 시설물 (FA-01) 무조건 SIDE 뷰 우선
        # -> 최신이 TOP이고, 이전이 SIDE라면 SIDE(이전)의 결과를 강제로 가져옴
        if latest_angle == "TOP_VIEW" and prev_angle == "SIDE_VIEW":
            latest_facilities = latest_result.get("facility_status", {}).get("facility_status", [])
            prev_facilities = prev_result.get("facility_status", {}).get("facility_status", [])
            
            # 이전 결과에서 FA-01의 상태를 추출
            prev_side_status = {f["zone"]: f["status"] for f in prev_facilities if f["zone"] in self.SIDE_DEPENDENT_FACILITIES}
            
            for f in latest_facilities:
                if f["zone"] in prev_side_status:
                    f["status"] = prev_side_status[f["zone"]]

        # [앙상블 룰 1'] 위쪽 뷰 필수 시설물 (FA-03)
        if latest_angle == "SIDE_VIEW" and prev_angle == "TOP_VIEW":
            latest_facilities = latest_result.get("facility_status", {}).get("facility_status", [])
            prev_facilities = prev_result.get("facility_status", {}).get("facility_status", [])
            
            # 이전 결과에서 FA-03의 상태를 추출
            prev_top_status = {f["zone"]: f["status"] for f in prev_facilities if f["zone"] in self.TOP_DEPENDENT_FACILITIES}
            
            for f in latest_facilities:
                if f["zone"] in prev_top_status:
                    f["status"] = prev_top_status[f["zone"]]

        # [앙상블 룰 2] 폭파구 / 불발탄 탐지 (신뢰도 기반 롤백)
        # -> 기본적으로 최신 영상을 따라가되, 최신 영상의 confidence가 기준치 미만이고
        # 이전 영상의 confidence가 더 높다면 이전 영상의 탐지 결과를 복원함
        def _swap_low_conf_objects(target_key, conf_category):
            # 1. 딕셔너리 구조 가져오기 (예: {"mission_code": "...", "crater_detect": [...]})
            latest_dict = latest_result.get(target_key, {})
            prev_dict = prev_result.get(target_key, {})
            
            # 2. 실제 타겟 리스트 추출 (두 번 파고들기)
            latest_items = latest_dict.get(target_key, [])
            prev_items = prev_dict.get(target_key, [])
            
            # 이전 영상 데이터들을 구역(zone)을 키(key)로 하는 딕셔너리로 변환 (빠른 검색용)
            prev_items_dict = {item["zone"]: item for item in prev_items}
            
            # confidence 리스트도 구역(zone)을 키로 매핑
            latest_confs = {item["zone"]: item.get("confidence", 1.0) for item in latest_conf.get(conf_category, [])}
            prev_confs = {item["zone"]: item.get("confidence", 1.0) for item in prev_conf.get(conf_category, [])}
            
            for i, l_item in enumerate(latest_items):
                zone = l_item["zone"]
                l_conf = latest_confs.get(zone, 1.0)
                
                # 1. 객체가 있고, 그 confidence가 기준치보다 낮다면?
                if l_conf < self.conf_threshold:
                    p_conf = prev_confs.get(zone, 0.0)
                    
                    # 2. 히스토리에서 그 영역에 같은 종류가 존재하고, 이전 confidence가 더 높다면 교환!
                    if zone in prev_items_dict and p_conf > l_conf:
                        prev_item = prev_items_dict[zone]

                        # JSON 구조상 폭파구는 'size', 불발탄은 'type' 키를 사용하므로 동적 추출
                        l_class = l_item.get("size", l_item.get("type", "unknown"))
                        p_class = prev_item.get("size", prev_item.get("type", "unknown"))
                        
                        # 교환 발생 로그 출력
                        print(f"[EnsembleManager] 🔄 낮은 confidence로 인한 객체 교환 ({conf_category.upper()}) | "
                              f"구역: {zone} | "
                              f"분류: {l_class} -> {p_class} | "
                              f"Confidence: {l_conf:.3f} -> {p_conf:.3f}")
                        
                        latest_items[i] = copy.deepcopy(prev_item)
                        
                    
                    # 3. 없다면? 어쩔 수 없이 놔둠 (현재 l_item 유지)
            latest_dict[target_key] = latest_items
            return latest_items

        # 폭파구 핀포인트 교체 (latest_dict가 latest_result["crater_detect"]를 그대로 참조하므로
        # mission_code를 포함한 {"mission_code":..., "crater_detect":[...]} 구조가 그대로 유지됨)
        _swap_low_conf_objects("crater_detect", "crater")
        # 불발탄 핀포인트 교체
        _swap_low_conf_objects("uxo_detect", "uxo")

        return latest_result


