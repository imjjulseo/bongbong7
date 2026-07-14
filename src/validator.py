# -*- coding: utf-8 -*-
"""
validator.py
============
JSON을 서버로 전송하기 전, 사람이 놓치기 쉬운 실수를 자동으로 잡아내는 모듈입니다.
- 시설물 6개 슬롯이 모두 있는가?
- crater_count와 crater_detect의 개수가 서로 일치하는가?
- 필수 필드가 비어있지 않은가?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc


class ValidationError(Exception):
    pass


def validate_all(outputs: dict, raise_on_error: bool = False):
    """
    outputs: {"start":..., "crater_detect":..., "crater_count":..., "runway_status":...,
              "facility_status":..., "uxo_detect":..., "uxo_count":..., "report":...}
    반환: {"ok": bool, "errors": [str,...], "warnings": [str,...]}
    """
    errors = []
    warnings = []

    # 1. 미션코드 일관성
    codes = set()
    for key, doc in outputs.items():
        if isinstance(doc, dict) and "mission_code" in doc:
            codes.add(doc["mission_code"])
    if len(codes) > 1:
        errors.append(f"파일마다 mission_code가 다릅니다: {codes}")
    if len(codes) == 1 and list(codes)[0] != fc.MISSION_CODE:
        warnings.append(f"mission_code가 field_config.MISSION_CODE와 다릅니다: {codes}")

    # 2. crater_count == crater_detect 총 개수
    if "crater_detect" in outputs and "crater_count" in outputs:
        detect_total = outputs["crater_detect"].get("crater_count_total", None)
        runway_count = outputs["crater_count"].get("runway_crater_count", None)
        if detect_total is not None and runway_count is not None and runway_count > detect_total:
            errors.append(
                f"활주로 폭파구 개수({runway_count})가 전체 탐지 개수({detect_total})보다 많을 수 없습니다."
            )

    # 3. 시설물 6개 슬롯 확인 (누락 방지 핵심 체크)
    if "facility_status" in outputs:
        facilities = outputs["facility_status"].get("facilities", [])
        slots_present = {f["slot"] for f in facilities}
        missing = set(fc.FACILITY_SLOTS) - slots_present
        if missing:
            errors.append(f"시설물 슬롯 누락: {sorted(missing)}")
        unconfirmed = [f["slot"] for f in facilities if f.get("status") == "미확인"]
        if unconfirmed:
            warnings.append(f"미확인 상태인 시설물: {unconfirmed} (재정찰 권장)")

    # 4. uxo_count <= uxo_detect 총 개수
    if "uxo_detect" in outputs and "uxo_count" in outputs:
        detect_total = outputs["uxo_detect"].get("uxo_count_total", None)
        runway_count = outputs["uxo_count"].get("runway_uxo_count", None)
        if detect_total is not None and runway_count is not None and runway_count > detect_total:
            errors.append(
                f"활주로 불발탄 개수({runway_count})가 전체 탐지 개수({detect_total})보다 많을 수 없습니다."
            )

    # 5. 활주로 가용길이 범위 체크 (0 ~ 필드 전체길이*스케일 이내인지)
    if "runway_status" in outputs:
        max_possible_m = fc.FIELD_WIDTH_CM * fc.REAL_METERS_PER_MODEL_CM
        avail = outputs["runway_status"].get("runway_available_length_m", None)
        if avail is not None and not (0 <= avail <= max_possible_m):
            errors.append(f"활주로 가용길이({avail}m)가 물리적으로 불가능한 범위입니다.")

    # 6. report.json 텍스트 비어있는지
    if "report" in outputs:
        text = outputs["report"].get("report_text", "")
        if not text or len(text.strip()) < 5:
            errors.append("report_text가 비어있거나 너무 짧습니다.")

    ok = len(errors) == 0
    if raise_on_error and not ok:
        raise ValidationError("; ".join(errors))

    return {"ok": ok, "errors": errors, "warnings": warnings}
