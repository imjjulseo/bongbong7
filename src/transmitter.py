# -*- coding: utf-8 -*-
"""
transmitter.py
===============
6단계: 완성된 8개 JSON을 대시보드로 순차/중복 전송하는 모듈입니다.

실제 서버 API(2026-07-14 현장 테스트로 확인):
  - URL: {fc.DASHBOARD_ENDPOINT_URL}/{미션코드}  (미션코드가 URL 경로에 포함됨)
  - 방식: multipart/form-data 파일 첨부 (raw JSON body가 아님!)
    필드명 "file", 파일명은 반드시 "start.json", "crater_detect.json" 등
    fc.TRANSMIT_ORDER의 키 + ".json" 형식이어야 서버가 어떤 보고인지 인식함.

fc.DASHBOARD_ENDPOINT_URL이 비어있으면(None) 전송 없이 스텁 로그만 남깁니다.

전송 정책:
  - 순차 전송: fc.TRANSMIT_ORDER(8개 JSON) 순서를 지켜 하나씩 전송
  - 중복 전송: 수신 유실 대비, 동일 JSON을 fc.TRANSMIT_DUPLICATE_COUNT회 반복 전송
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
import field_config as fc

import requests

_UNSET = object()  # endpoint 인자를 "안 넘김"(설정값 사용)과 "명시적 None"(전송 안 함)을 구분하기 위한 센티널


def _send_one(key: str, payload: dict, url: str, timeout: float):
    """단일 JSON을 파일(예: start.json)로 fc.TRANSMIT_DUPLICATE_COUNT회 중복 업로드. 각 시도 결과를 기록."""
    filename = f"{key}.json"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    attempts = []
    for attempt in range(1, max(1, fc.TRANSMIT_DUPLICATE_COUNT) + 1):
        try:
            resp = requests.post(
                url,
                files={"file": (filename, body, "application/json")},
                timeout=timeout,
            )
            attempts.append({
                "attempt": attempt, "ok": resp.ok, "status_code": resp.status_code,
            })
        except Exception as e:
            attempts.append({"attempt": attempt, "ok": False, "error": str(e)})
    return {"key": key, "sent": True, "attempts": attempts}


def transmit_all(outputs: dict, endpoint: str = _UNSET, order: list = None, timeout: float = None) -> dict:
    """
    outputs: {"start":..., "crater_detect":..., ..., "report":...} (pipeline.run()의 outputs)
    endpoint: 생략 시 fc.DASHBOARD_ENDPOINT_URL 사용(미션코드 경로는 자동으로 붙음).
              명시적으로 None을 넘기면 실제 전송 없이 스텁 로그만 남김.

    반환: {"endpoint_configured": bool, "results": [{"key":..., "sent": bool, ...}, ...]}
    """
    endpoint = fc.DASHBOARD_ENDPOINT_URL if endpoint is _UNSET else endpoint
    order = order or fc.TRANSMIT_ORDER
    timeout = timeout if timeout is not None else fc.TRANSMIT_TIMEOUT_SEC

    results = []
    if not endpoint:
        # TODO: 대시보드 API 엔드포인트가 확정되면 fc.DASHBOARD_ENDPOINT_URL을 채우세요.
        for key in order:
            if key not in outputs:
                continue
            results.append({"key": key, "sent": False, "reason": "DASHBOARD_ENDPOINT_URL 미설정(스텁)"})
        return {"endpoint_configured": False, "results": results}

    url = f"{endpoint.rstrip('/')}/{fc.MISSION_CODE}"

    for key in order:
        if key not in outputs:
            continue
        results.append(_send_one(key, outputs[key], url, timeout))

    return {"endpoint_configured": True, "results": results}
