# -*- coding: utf-8 -*-
"""
report_generator.py
====================
5단계: LLM API 키가 제공되지 않으므로, 로컬에서 실행되는 LLM(Ollama 등)을 사용합니다.
Ollama는 http://localhost:11434 에서 REST API를 제공하며, 이는 '외부 클라우드 서비스'가
아니라 '내 컴퓨터에서 도는 프로그램'이므로 대회 규정(외부 GPU서버/클라우드AI서비스 금지)에
저촉되지 않습니다. (단, 사전에 운영진에게 로컬 LLM 사용 가능 여부를 반드시 재확인하세요.)

글자수 제약: report_text는 반드시 REPORT_MIN_CHARS~REPORT_MAX_CHARS(기본 50~100자) 이내여야
합니다(대회 규정). 프롬프트에 명시하고, LLM이 규정을 넘겨 생성해도 _enforce_length()가
자연스러운 위치(공백)에서 잘라 강제로 100자 이내로 맞춥니다.

안전장치: 로컬 LLM 서버가 응답하지 않을 경우(설치 안됨/타임아웃 등),
자동으로 '템플릿 기반 오프라인 보고서 생성'으로 폴백합니다.
-> 이러면 LLM이 죽어도 임무는 절대 실패하지 않습니다.
"""
import json
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"   # 대회 PC 사양에 맞춰 사전에 벤치마크 후 결정
REQUEST_TIMEOUT_SEC = 8        # 임무 시간이 촉박하므로 타임아웃을 짧게 설정


REPORT_SYSTEM_PROMPT = f"""당신은 공군 전투피해평가(BDA) 보고서를 작성하는 참모 장교입니다.
아래 JSON 탐지 데이터를 바탕으로 활주로 가용 여부, 폭파구/불발탄 개수, 시설물 피해 현황 중
가장 중요한 내용만 압축한 한국어 상황보고 한 문장을 작성하세요.
반드시 {fc.REPORT_MIN_CHARS}자 이상 {fc.REPORT_MAX_CHARS}자 이내로 작성하세요.
반드시 한국어 문장만 출력하고, 다른 설명이나 따옴표는 붙이지 마세요."""


def _build_prompt(summary: dict) -> str:
    return (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"[탐지 데이터]\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
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
    """
    prompt = _build_prompt(summary)
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
    return _enforce_length(text)


def generate_report_offline_template(summary: dict) -> str:
    """
    LLM 없이도 항상 동작하는 결정론적(deterministic) 보고서 생성기.
    이게 있어야 LLM 서버 장애 시에도 임무가 100% 완주됩니다.
    50~100자 제약에 맞춰 한 문장으로 압축합니다.
    """
    crater_total = summary.get("crater_count_total", 0)
    runway_crater = summary.get("runway_crater_count", 0)
    runway_available_m = summary.get("runway_available_length_m", 0)
    facility_damage = summary.get("facility_damage_summary", {})
    uxo_total = summary.get("uxo_count_total", 0)
    uxo_runway = summary.get("uxo_runway_count", 0)

    destroy = facility_damage.get("destroy", 0)
    fire = facility_damage.get("fire", 0)
    unconfirmed = facility_damage.get("unconfirmed", 0)

    runway_usable = "가능" if runway_available_m >= 300 else ("제한 가능" if runway_available_m > 0 else "불가")

    text = (
        f"활주로 가용길이 {runway_available_m}m로 이착륙 {runway_usable}, "
        f"폭파구 {crater_total}개(활주로 {runway_crater}), 불발탄 {uxo_total}개(활주로 {uxo_runway}), "
        f"시설물 파손 {destroy}·화재 {fire}·미확인 {unconfirmed}건."
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
