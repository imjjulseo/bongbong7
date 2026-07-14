# -*- coding: utf-8 -*-
"""
img_io.py
=========
Windows에서 cv2.imread/imwrite는 절대경로에 비-ASCII(한글 등) 문자가 섞이면
파일을 읽거나 쓰지 못합니다(OpenCV가 내부적으로 로컬 코드페이지로 경로를 처리하는
한계 때문). 이 프로젝트 경로 자체에 한글이 포함되어 있어(예: "해커톤 본선"),
np.fromfile/tofile + cv2.imdecode/imencode로 우회한 안전한 대체 함수를 제공합니다.
"""
import os
import cv2
import numpy as np


def imread_safe(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread와 동일하게 동작하되, 비-ASCII 경로에서도 안전하게 읽습니다.
    읽기에 실패하면(파일 없음 등) cv2.imread와 마찬가지로 None을 반환합니다."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_safe(path, img, params=None):
    """cv2.imwrite와 동일하게 동작하되, 비-ASCII 경로에서도 안전하게 씁니다.
    params: cv2.imwrite의 인코딩 파라미터(예: [cv2.IMWRITE_PNG_COMPRESSION, 0])와 동일한 형식.
    성공 여부를 bool로 반환합니다."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True
