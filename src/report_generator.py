# -*- coding: utf-8 -*-
"""
report_generator.py
====================
5단계: 생성형 AI(LLM)로 정찰 결과를 종합해 상황보고서 한 문장을 자동 생성합니다.
대회 규정상 사용 모델 종류에는 제한이 없으나(현장 안내 기준), 인터넷이 불안정한 대회 현장
환경을 고려해 기본값은 로컬 Ollama(http://localhost:11434)로 둡니다.

대회 규정(안내받은 내용 기준):
  - 탐지 결과를 기반으로 자동 생성해야 하며, 확인되지 않은 내용은 포함하지 않음.
  - 보고 시각(년/월/일/시/분)을 포함해야 함.
  - 활주로구역 내 탐지된 폭파구 총 개수를 포함해야 함.
  - 활주로 가용길이(폭파구 탐지미션 규칙에 따라 산출)를 포함해야 함.
  - 가용길이에 따른 상태/운용여부(fc.classify_runway_status 기준표)를 포함해야 함.
  - 공백 포함 100자 이내.

상태/운용여부/가용길이/폭파구 개수/보고시각은 코드에서 미리 정확히 계산해 LLM에 "이미 확정된
사실"로 넘깁니다 - LLM은 그 사실을 규정된 글자수 안에서 자연스러운 한국어 문장으로 표현하는
역할만 담당합니다(LLM이 수치를 스스로 계산/추측하게 하면 오류 위험이 있으므로).

안전장치: LLM 서버가 응답하지 않을 경우(설치 안됨/타임아웃 등) 자동으로 결정론적 템플릿
생성으로 폴백합니다 -> LLM이 죽어도 임무는 절대 실패하지 않습니다.
"""
import json
import requests
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"   # 대회 PC 사양에 맞춰 사전에 벤치마크 후 결정
REQUEST_TIMEOUT_SEC = 8        # 임무 시간이 촉박하므로 타임아웃을 짧게 설정


def _format_report_time(dt: datetime = None) -> str:
    dt = dt or datetime.now()
    return f"{dt.year}년 {dt.month:02d}월 {dt.day:02d}일 {dt.hour:02d}시 {dt.minute:02d}분"


def _format_length_m(length_m) -> str:
    """정수면 소수점 없이(예: 1200m), 아니면 소수 첫째자리까지(예: 1200.5m) 표기."""
    return str(int(length_m)) if float(length_m) == int(length_m) else str(round(float(length_m), 1))


def _prepare_facts(summary: dict) -> dict:
    """summary(내부 집계 결과)로부터 보고서에 반드시 들어가야 할 확정 사실들을 계산."""
    runway_available_m = summary.get("runway_available_length_m", 0)
    status, availability = fc.classify_runway_status(runway_available_m)
    return {
        "report_time": _format_report_time(),
        "runway_crater_count": summary.get("runway_crater_count", 0),
        "runway_available_length_m": runway_available_m,
        "runway_status": status,
        "runway_availability": availability,
    }


REPORT_SYSTEM_PROMPT = f"""당신은 공군 전투피해평가(BDA) 보고서를 작성하는 참모 장교입니다.
아래 [확정 사실]에 적힌 내용만 사용해서 한국어 상황보고 한 문장을 작성하세요.
- 반드시 포함할 내용: 보고 시각, 활주로구역 폭파구 개수, 활주로 가용길이, 상태, 운용여부.
- "폭파구"는 반드시 이 두 글자 그대로 쓰세요("폭발구", "파괴구" 등 다른 표현으로 바꾸지 마세요).
- runway_status, runway_availability의 핵심 문구 자체는 절대 바꾸지 말고 원문 그대로 넣으세요.
  문장을 자연스럽게 끝맺기 위해 뒤에 서술어를 붙이는 것은 괜찮습니다.
  단, "사용 가능 여부 검토"에는 반드시 "~가 필요합니다"를 붙여
  "사용 가능 여부 검토가 필요합니다"처럼 쓰세요 ("~검토 중입니다"는 쓰지 마세요).
- [확정 사실]에 없는 내용(시설물 피해, 불발탄 등 확인되지 않은 정보)은 절대 추가하지 마세요.
- 공백 포함 {fc.REPORT_MAX_CHARS}자를 절대 넘기지 마세요.
- 반드시 한국어 문장만 출력하고, 다른 설명이나 따옴표는 붙이지 마세요."""


def _build_prompt(facts: dict) -> str:
    return (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"[확정 사실]\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        f"[보고서]"
    )


def _enforce_length(text: str) -> str:
    """report_text 글자수 제약(fc.REPORT_MIN_CHARS~REPORT_MAX_CHARS)을 강제 적용.
    100자를 넘으면 마지막 공백 기준으로 자연스럽게 잘라낸다(단어 중간 절단 방지)."""
    text = text.strip()
    if len(text) > fc.REPORT_MAX_CHARS:
        # 마침표를 붙일 자리(1자)를 남겨두고 자르기 -> 최종 길이가 REPORT_MAX_CHARS를 넘지 않도록 보장
        truncated = text[:fc.REPORT_MAX_CHARS - 1]
        if " " in truncated:
            truncated = truncated[:truncated.rfind(" ")]
        text = truncated.rstrip(",.· ") + "."
    return text


def generate_report_via_local_llm(summary: dict, model: str = OLLAMA_MODEL) -> str:
    """
    로컬 Ollama 서버에 요청. 실패하면 예외를 던짐 (호출부에서 폴백 처리).
    LLM이 "폭파구"를 "폭발구"로 오타내거나 상태/운용여부 문구를 임의로 바꿔버릴 위험이
    있으므로, 생성된 문장에 이 핵심 용어들이 원문 그대로 포함되어 있는지 검증하고 - 아니면
    규정 위반/오타로 보고 예외를 던져 템플릿으로 폴백시킨다.
    """
    facts = _prepare_facts(summary)
    prompt = _build_prompt(facts)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},  # 보고서이므로 창의성보다 일관성 우선
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("로컬 LLM이 빈 응답을 반환했습니다.")
    text = _enforce_length(text)
    if "폭파구" not in text:
        raise RuntimeError(f"LLM이 '폭파구' 용어를 그대로 쓰지 않음: {text!r}")
    if facts["runway_status"] not in text or facts["runway_availability"] not in text:
        raise RuntimeError(
            f"LLM이 규정 문구를 그대로 쓰지 않음(상태/운용여부 누락 또는 변형): {text!r}"
        )
    # 정답 문구가 있어도, 다른 등급의 상태/운용여부 문구까지 같이 섞여 있으면 등급 혼동이므로 거부.
    # (사용 가능/제한적 사용 가능은 서로 부분 문자열 관계라 이 검사에서 제외)
    for _, other_status, other_availability in fc.RUNWAY_STATUS_THRESHOLDS:
        if other_status != facts["runway_status"] and other_status in text:
            raise RuntimeError(f"LLM이 다른 등급의 상태 문구를 섞어씀: {text!r}")
        if (other_availability != facts["runway_availability"]
                and other_availability not in ("사용 가능", "제한적 사용 가능")
                and other_availability in text):
            raise RuntimeError(f"LLM이 다른 등급의 운용여부 문구를 섞어씀: {text!r}")
    return text


def generate_report_offline_template(summary: dict) -> str:
    """
    LLM 없이도 항상 동작하는 결정론적(deterministic) 보고서 생성기.
    이게 있어야 LLM 서버 장애 시에도 임무가 100% 완주됩니다.
    대회 규정(보고시각/활주로 폭파구 개수/가용길이/상태/운용여부, 100자 이내)을 그대로 충족합니다.
    """
    facts = _prepare_facts(summary)
    text = (
        f"{facts['report_time']} 기준, 활주로 폭파구 {facts['runway_crater_count']}개 탐지, "
        f"가용길이 {_format_length_m(facts['runway_available_length_m'])}m, "
        f"{facts['runway_status']}·{facts['runway_availability']}."
    )
    return _enforce_length(text)


def generate_report(summary: dict, use_llm: bool = True) -> dict:
    """
    메인 진입점. LLM 시도 -> 실패 시 템플릿 폴백.
    반환: {"text": str, "source": "local_llm" | "offline_template"}
    """
    if use_llm:
        try:
            text = generate_report_via_local_llm(summary)
            return {"text": text, "source": "local_llm"}
        except Exception as e:
            # 네트워크 차단/서버 미실행/타임아웃 등 어떤 이유든 조용히 폴백
            fallback_text = generate_report_offline_template(summary)
            return {"text": fallback_text, "source": f"offline_template (llm_error: {e})"}
    else:
        text = generate_report_offline_template(summary)
        return {"text": text, "source": "offline_template"}
