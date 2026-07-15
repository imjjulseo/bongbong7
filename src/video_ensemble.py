
from collections import deque
import copy

class VideoEnsembleManager:
    def __init__(self, max_history=2):
        # 최근 N개의 영상 처리 결과만 유지 (과거 정보 과의존 방지)
        self.history = deque(maxlen=max_history)
        
        # 지붕에 가려져 측면(Side) 뷰가 필수적인 특정 시설물 지정
        self.SIDE_DEPENDENT_FACILITIES = ["FA-01"] 

    def add_video_result(self, json_outputs, angle_type):
        """새로운 영상의 JSON 결과와 판별된 각도를 히스토리에 추가"""
        self.history.append({
            "data": json_outputs,
            "angle": angle_type
        })

    def get_ensembled_result(self):
        """히스토리에 쌓인 결과를 바탕으로 최적의 JSON 생성"""
        if not self.history:
            return None
            
        # 가장 최근 결과를 베이스로 복사
        latest_result = copy.deepcopy(self.history[-1]["data"])
        latest_angle = self.history[-1]["angle"]
        
        # 히스토리가 1개뿐이면 앙상블 없이 그대로 반환
        if len(self.history) == 1:
            return latest_result
            
        prev_result = self.history[0]["data"]
        prev_angle = self.history[0]["angle"]

        # ---------------------------------------------------------
        # [앙상블 룰 1] 측면 뷰 필수 시설물 (예: FA-01)
        # ---------------------------------------------------------
        # 만약 최근 영상이 TOP인데, 이전 영상이 SIDE 였다면?
        # -> 다른 시설물은 TOP(최근)을 쓰되, FA-01만큼은 SIDE(이전)의 결과를 가져옴
        if latest_angle == "TOP_VIEW" and prev_angle == "SIDE_VIEW":
            # facility_status.json 병합
            latest_facilities = latest_result.get("facility_status", {}).get("facilities", [])
            prev_facilities = prev_result.get("facility_status", {}).get("facilities", [])
            
            # 이전 결과에서 FA-1의 상태를 찾아서 현재 결과에 덮어쓰기
            prev_fa01_status = next((f["status"] for f in prev_facilities if f["zone"] in self.SIDE_DEPENDENT_FACILITIES), None)
            
            if prev_fa01_status:
                for f in latest_facilities:
                    if f["zone"] in self.SIDE_DEPENDENT_FACILITIES:
                        f["status"] = prev_fa01_status

        # ---------------------------------------------------------
        # [앙상블 룰 2] 폭파구 / 불발탄 탐지 (Top View 우선)
        # ---------------------------------------------------------
        # 최근 영상이 SIDE라서 객체 탐지가 엉망이 되었을 가능성이 높고,
        # 이전 영상이 TOP이라서 탐지가 잘 되었다면?
        # -> Crater와 UXO 리스트는 이전 영상(TOP)의 데이터를 그대로 복원!
        if latest_angle == "SIDE_VIEW" and prev_angle == "TOP_VIEW":
            latest_result["crater_detect"] = prev_result["crater_detect"]
            latest_result["uxo_detect"] = prev_result["uxo_detect"]
            latest_result["crater_count"] = prev_result["crater_count"]
            latest_result["uxo_count"] = prev_result["uxo_count"]

        return latest_result