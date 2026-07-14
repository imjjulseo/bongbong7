# -*- coding: utf-8 -*-
"""
schemas.py
==========
대회에서 요구하는 8개 전송 JSON 파일(mission_code 표 기준)의 '틀'을 정의합니다.
파일명 <-> 임무명 매핑:
  1. start.json            준비단계 (미션코드)
  2. crater_detect.json    폭파구 크기/위치 탐지
  3. crater_count.json     활주로 폭파구 개수
  4. runway_status.json    활주로 가용길이
  5. facility_status.json  시설물 상태
  6. uxo_detect.json       불발탄 위치 및 종류
  7. uxo_count.json        활주로 불발탄 개수
  8. report.json           LLM 기반 작전상황 보고
"""
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_start_json(mission_code: str) -> dict:
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
    }


def build_crater_detect_json(mission_code: str, craters: list) -> dict:
    """
    craters: [{"id": str, "segment": str, "size_class": "대형|중형|소형",
               "center_world_cm": [x,y], "diameter_m": float}, ...]
    """
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "crater_count_total": len(craters),
        "craters": craters,
    }


def build_crater_count_json(mission_code: str, runway_crater_count: int) -> dict:
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "runway_crater_count": runway_crater_count,
    }


def build_runway_status_json(mission_code: str, longest_segment: dict,
                              blocked_segments: list, available_length_m: float) -> dict:
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "blocked_segments": blocked_segments,
        "longest_available_run": longest_segment,   # {"segments":[...], "length_m": float}
        "runway_available_length_m": available_length_m,
        "runway_usable": available_length_m > 0,
    }


def build_facility_status_json(mission_code: str, facilities: list) -> dict:
    """
    facilities: [{"slot": "FA-01", "type": "관제탑", "status": "정상|파손|화재|미확인"}, ...]
    반드시 6개 슬롯이 모두 존재해야 함 (누락 방지)
    """
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "facility_count": len(facilities),
        "facilities": facilities,
    }


def build_uxo_detect_json(mission_code: str, uxos: list) -> dict:
    """
    uxos: [{"id": str, "segment": str, "type": "미사일|포탄|자탄",
            "center_world_cm": [x,y], "confidence": float}, ...]
    """
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "uxo_count_total": len(uxos),
        "uxos": uxos,
    }


def build_uxo_count_json(mission_code: str, runway_uxo_count: int) -> dict:
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "runway_uxo_count": runway_uxo_count,
    }


def build_report_json(mission_code: str, report_text: str, summary: dict) -> dict:
    return {
        "mission_code": mission_code,
        "timestamp": _now_iso(),
        "report_text": report_text,
        "summary": summary,
    }
