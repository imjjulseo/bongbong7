# -*- coding: utf-8 -*-
"""
schemas.py
==========
대회 측이 제공한 8개 전송 JSON 템플릿 형식 그대로 '틀'을 정의합니다.
파일명 <-> 임무명 매핑:
  1. start.json            준비단계 (미션코드)
  2. crater_detect.json    폭파구 크기/위치 탐지
  3. crater_count.json     활주로 폭파구 개수
  4. runway_status.json    활주로 가용길이 (cm)
  5. facility_status.json  시설물 상태
  6. uxo_detect.json       불발탄 위치 및 종류
  7. uxo_count.json        활주로 불발탄 개수
  8. report.json           LLM 기반 작전상황 보고

주의: timestamp 등 부가 필드는 넣지 않습니다(템플릿에 없는 필드는 전부 제외).
"""


def build_start_json(mission_code: str) -> dict:
    return {
        "mission_code": mission_code,
    }


def build_crater_detect_json(mission_code: str, craters: list) -> dict:
    """craters: [{"zone": str, "size": "big|medium|small"}, ...]"""
    return {
        "mission_code": mission_code,
        "crater_detect": craters,
    }


def build_crater_count_json(mission_code: str, crater_count: int) -> dict:
    return {
        "mission_code": mission_code,
        "crater_count": crater_count,
    }


def build_runway_status_json(mission_code: str, runway_status_cm) -> dict:
    """runway_status_cm: 활주로 가용길이 (실좌표 cm 단위)"""
    return {
        "mission_code": mission_code,
        "runway_status": runway_status_cm,
    }


def build_facility_status_json(mission_code: str, facilities: list) -> dict:
    """facilities: [{"zone": "FA-01", "status": "normal|destroy|fire"}, ...]
    반드시 6개 슬롯이 모두 존재해야 함 (누락 방지)"""
    return {
        "mission_code": mission_code,
        "facility_status": facilities,
    }


def build_uxo_detect_json(mission_code: str, uxos: list) -> dict:
    """uxos: [{"zone": str, "type": "missile|dumb|cluster"}, ...]"""
    return {
        "mission_code": mission_code,
        "uxo_detect": uxos,
    }


def build_uxo_count_json(mission_code: str, uxo_count: int) -> dict:
    return {
        "mission_code": mission_code,
        "uxo_count": uxo_count,
    }


def build_report_json(mission_code: str, report_text: str) -> dict:
    return {
        "mission_code": mission_code,
        "report": report_text,
    }
