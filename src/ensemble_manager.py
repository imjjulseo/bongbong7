import copy
from collections import deque


class VideoEnsembleManager:
    """최근 N개의 영상 파이프라인 결과를 유지하며 각도(Angle) 기반으로 결과를 융합합니다."""
    def __init__(self, max_history=2):
        self.history = deque(maxlen=max_history)
        # 측면 뷰(Side View) 판독이 필수적인 시설물 목록 (테스트 결과 반영)
        self.SIDE_DEPENDENT_FACILITIES = ["FA-01"] 

    def add_video_result(self, json_outputs, angle_type):
        """새로운 영상의 JSON 결과와 판별된 각도를 히스토리에 추가"""
        self.history.append({
            "data": json_outputs,
            "angle": angle_type
        })

    def get_ensembled_result(self):
        """히스토리에 쌓인 결과를 바탕으로 최적의 병합된 JSON 생성"""
        if not self.history:
            return None
            
        # 가장 최근 결과를 베이스로 깊은 복사
        latest_result = copy.deepcopy(self.history[-1]["data"])
        latest_angle = self.history[-1]["angle"]
        
        # 히스토리가 1개뿐이면 앙상블 없이 그대로 반환
        if len(self.history) == 1:
            return latest_result
        
        prev_result = self.history[0]["data"]
        prev_angle = self.history[0]["angle"]

        # [앙상블 룰 1] 측면 뷰 필수 시설물 (FA-01 등)
        # -> 최근 영상이 TOP인데, 이전 영상이 SIDE 였다면 SIDE(이전)의 결과를 유지
        if latest_angle == "TOP_VIEW" and prev_angle == "SIDE_VIEW":
            latest_facilities = latest_result.get("facility_status", {}).get("facilities", [])
            prev_facilities = prev_result.get("facility_status", {}).get("facilities", [])
            
            # 이전 결과에서 특정 시설물의 상태를 추출
            prev_side_status = {f["zone"]: f["status"] for f in prev_facilities if f["zone"] in self.SIDE_DEPENDENT_FACILITIES}
            
            for f in latest_facilities:
                if f["zone"] in prev_side_status:
                    f["status"] = prev_side_status[f["zone"]]

        # [앙상블 룰 2] 폭파구 / 불발탄 탐지 (Top View 절대 우선)
        # -> 최근 영상이 SIDE라서 객체 탐지가 부정확할 가능성이 높고, 이전 영상이 TOP이라면
        # Crater와 UXO 탐지 결과는 이전 영상(TOP)의 데이터를 덮어씌워 복원
        if latest_angle == "SIDE_VIEW" and prev_angle == "TOP_VIEW":
            for key in ["crater_detect", "uxo_detect", "crater_count", "uxo_count", "runway_status"]:
                if key in prev_result:
                    latest_result[key] = prev_result[key]

        return latest_result


