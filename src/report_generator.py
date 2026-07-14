# -*- coding: utf-8 -*-
"""
report_generator.py
====================
LLM API 키가 제공되지 않으므로, 로컬에서 실행되는 LLM(Ollama 등)을 사용합니다.
Ollama는 http://localhost:11434 에서 REST API를 제공하며, 이는 '외부 클라우드 서비스'가
아니라 '내 컴퓨터에서 도는 프로그램'이므로 대회 규정(외부 GPU서버/클라우드AI서비스 금지)에
저촉되지 않습니다. (단, 사전에 운영진에게 로컬 LLM 사용 가능 여부를 반드시 재확인하세요.)

안전장치: 로컬 LLM 서버가 응답하지 않을 경우(설치 안됨/타임아웃 등),
자동으로 '템플릿 기반 오프라인 보고서 생성'으로 폴백합니다.
-> 이러면 LLM이 죽어도 임무는 절대 실패하지 않습니다.
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"   # 대회 PC 사양에 맞춰 사전에 벤치마크 후 결정
REQUEST_TIMEOUT_SEC = 8        # 임무 시간이 촉박하므로 타임아웃을 짧게 설정


# JSON 형식을 강제하기 위한 프롬프트. Ollama의 format="json" 옵션과 함께 사용하면
# 문법 제약(GBNF와 유사한 효과)으로 스키마 이탈을 크게 줄일 수 있습니다.
REPORT_SYSTEM_PROMPT = """당신은 공군 전투피해평가(BDA) 보고서를 작성하는 참모 장교입니다.
아래 JSON 데이터를 바탕으로, 다음 3단계 판단을 포함한 한국어 상황보고서를 작성하세요:
1) 물리적 피해 현황 (폭파구/시설물/불발탄 개수 및 위치)
2) 기능적 피해 판단 (활주로 이착륙 가능 여부, 가용길이 기준)
3) 작전 권고사항 (우선 복구 대상, 불발탄 처리반 투입 필요 구역)
보고서는 5문장 이내로 간결하게 작성하세요. 반드시 한국어로만 작성하세요."""


def _build_prompt(summary: dict) -> str:
    return (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"[탐지 데이터]\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"[보고서]"
    )


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
    return text


def generate_report_offline_template(summary: dict) -> str:
    """
    LLM 없이도 항상 동작하는 결정론적(deterministic) 보고서 생성기.
    이게 있어야 LLM 서버 장애 시에도 임무가 100% 완주됩니다.
    """
    crater_total = summary.get("crater_count_total", 0)
    runway_available_m = summary.get("runway_available_length_m", 0)
    facility_damage = summary.get("facility_damage_summary", {})
    uxo_total = summary.get("uxo_count_total", 0)
    uxo_runway = summary.get("uxo_runway_count", 0)

    runway_usable = "가능" if runway_available_m >= 300 else ("제한적 가능" if runway_available_m > 0 else "불가")

    damaged = facility_damage.get("파손", 0)
    fire = facility_damage.get("화재", 0)
    unconfirmed = facility_damage.get("미확인", 0)

    lines = []
    lines.append(
        f"활주로 및 유도로 정찰 결과 총 {crater_total}개의 폭파구가 식별되었으며, "
        f"장애구간을 제외한 최장 가용 활주로 길이는 약 {runway_available_m}m로 산출됨."
    )
    lines.append(f"현 가용길이 기준 항공기 이착륙은 {runway_usable}할 것으로 판단됨.")
    if fire > 0 or damaged > 0:
        lines.append(f"시설물 중 화재 {fire}개소, 파손 {damaged}개소가 확인되어 우선 복구가 필요함.")
    else:
        lines.append("주요 시설물은 대부분 정상 상태로 확인됨.")
    if unconfirmed > 0:
        lines.append(f"단, {unconfirmed}개 시설물은 영상 미확보로 상태 미확인 상태이며 재정찰이 필요함.")
    if uxo_total > 0:
        lines.append(
            f"불발탄 {uxo_total}개(활주로 구간 {uxo_runway}개 포함)가 탐지되어 "
            f"폭발물처리반(EOD) 투입이 요구됨."
        )
    else:
        lines.append("불발탄은 식별되지 않았음.")

    return " ".join(lines)


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
