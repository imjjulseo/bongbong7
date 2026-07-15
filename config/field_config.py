# -*- coding: utf-8 -*-
"""
field_config.py
================
경기장의 모든 '고정된 사실'을 모아둔 파일입니다.
슬라이드에서 미리 공개된 정보(치수표, 구간 레이아웃, 축척)를 그대로 코드로 옮겼습니다.
대회 규정상 ArUco 마커와 경기장 레이아웃은 테스트 경기장 포함 사전 공개되므로,
이 파일의 좌표값은 실제 테스트 경기장에서 다시 측정해 업데이트해야 합니다.
"""

# ---------------------------------------------------------
# 1. 경기장 전체 규격
# ---------------------------------------------------------
FIELD_WIDTH_CM = 500       # 가로 (실제 경기장 모형 크기)
FIELD_HEIGHT_CM = 400      # 세로
SCALE_RATIO = 600          # 1 : 600 축척
# 모형 1cm = 실제 600cm = 실제 6m
REAL_METERS_PER_MODEL_CM = SCALE_RATIO / 100.0  # = 6.0 m/cm


# ---------------------------------------------------------
# 2. ArUco 마커 설정 (경기장 4개 모서리 기준점)
# ---------------------------------------------------------
# 2026-07-14 실측(바탕화면/data 드론 영상)으로 갱신됨: 실제 마커 ID는 0~3이 아니라 1~4이고,
# 코너 대응도 예상과 달랐음(각 마커가 FA-01/FA-03/FA-04/FA-06 코너에 위치).
# 마커 근접 crop으로 라벨을 직접 확인해 검증함 (id=1은 "FA-01" 텍스트, id=4는 "FA-06" 텍스트가
# 바로 옆에 촬영됨). id=2/3는 같은 변(TW-A측/TW-B측)에 있는 나머지 두 코너로 소거법 확정.
ARUCO_DICT_NAME = "DICT_4X4_50"

# 마커 ID -> 경기장 기준 좌표(cm).
ARUCO_MARKER_WORLD_POSITIONS = {
    1: (0.0, 0.0),                          # FA-01 코너
    2: (FIELD_WIDTH_CM, 0.0),               # FA-03 코너
    3: (0.0, FIELD_HEIGHT_CM),              # FA-04 코너
    4: (FIELD_WIDTH_CM, FIELD_HEIGHT_CM),   # FA-06 코너
}

# ArUco 마커의 4개 코너: 0: 좌상(TL), 1: 우상(TR), 2: 우하(BR), 3: 좌하(BL)
ARUCO_MARKER_CORNER_INDEX = {
    1: 0,  # ID 1번 마커는 우상단(TR) 코너를 경기장 모서리로 사용
    2: 1,  # ID 2번 마커는 우하단(BR) 코너를 경기장 모서리로 사용
    3: 3,  # ID 3번 마커는 좌하단(BL) 코너를 경기장 모서리로 사용
    4: 2, 
}

# ---------------------------------------------------------
# 3. 구역/구간 레이아웃 (이미지 18 기준)
#    좌표계: 원점(0,0)은 좌상단, x는 오른쪽, y는 아래쪽 (cm 단위, 모형 기준)
# ---------------------------------------------------------
def _row(y0, y1, names, widths, x0=0.0):
    """가로로 나열된 구간들의 (x_min,y_min,x_max,y_max) 딕셔너리를 생성하는 헬퍼"""
    segs = {}
    x = x0
    for name, w in zip(names, widths):
        segs[name] = {"x_min": x, "y_min": y0, "x_max": x + w, "y_max": y1}
        x += w
    return segs


SEGMENTS = {}

# 상단 시설물 구역 (FA-01, FA-02, FA-03) : y 0~80
SEGMENTS.update(_row(
    0, 80,
    ["FA-01", "FA-02", "FA-03"],
    [160, 180, 160],
))

# 유도로 A구역 (TW-A1~TW-A5) : y 80~160
SEGMENTS.update(_row(
    80, 160,
    ["TW-A1", "TW-A2", "TW-A3", "TW-A4", "TW-A5"],
    [100, 100, 100, 100, 100],
))

# 활주로 구역 (RW-01~RW-10) : y 160~240
SEGMENTS.update(_row(
    160, 240,
    [f"RW-{i:02d}" for i in range(1, 11)],
    [50] * 10,
))

# 유도로 B구역 (TW-B1~TW-B5) : y 240~320
SEGMENTS.update(_row(
    240, 320,
    ["TW-B1", "TW-B2", "TW-B3", "TW-B4", "TW-B5"],
    [100, 100, 100, 100, 100],
))

# 하단 시설물 구역 (FA-04, FA-05, FA-06) : y 320~400
SEGMENTS.update(_row(
    320, 400,
    ["FA-04", "FA-05", "FA-06"],
    [160, 180, 160],
))

# 활주로 구간 순서 (가용길이 산출용, 왼쪽->오른쪽)
RUNWAY_SEGMENT_ORDER = [f"RW-{i:02d}" for i in range(1, 11)]
TAXIWAY_A_ORDER = ["TW-A1", "TW-A2", "TW-A3", "TW-A4", "TW-A5"]
TAXIWAY_B_ORDER = ["TW-B1", "TW-B2", "TW-B3", "TW-B4", "TW-B5"]
FACILITY_SLOTS = ["FA-01", "FA-02", "FA-03", "FA-04", "FA-05", "FA-06"]

# 예시 검증: RW-01 실제 길이 = 50cm(모형) * 6.0(m/cm) = 300m -> 슬라이드 예시와 일치
_RW01_REAL_LEN = (SEGMENTS["RW-01"]["x_max"] - SEGMENTS["RW-01"]["x_min"]) * REAL_METERS_PER_MODEL_CM
assert abs(_RW01_REAL_LEN - 300.0) < 1e-6, "RW-01 실제 길이가 300m가 되어야 합니다(슬라이드 예시 기준)"


# ---------------------------------------------------------
# 4. 시설물 종류 (6종) - FA 슬롯과의 매핑
#    2026-07-14 실측 영상(바탕화면/data) 확인 결과 FA-01/FA-03은 아래처럼 확정.
#    나머지(FA-02,04,05,06)는 영상에서 뚜렷한 관제탑/무기고 형태가 안 보여 아직 미확정
#    placeholder임 - "건물 근접" 폴더 클로즈업으로 추가 확인 필요.
# ---------------------------------------------------------
FACILITY_TYPE_BY_SLOT = {
    "FA-01": "radar",           # 확인됨: 돔+격자타워, "FA-01" 라벨과 함께 촬영됨
    "FA-02": "building_2",      # 미확정: 소형 장비동(환기구), 관제레이더 아님(FA-01이 레이더임)
    "FA-03": "hangar",          # 확인됨: 대형 곡면지붕 구조물
    "FA-04": "building_3",      # 미확정: 일반 건물 형태
    "FA-05": "building_4",      # 미확정: 일반 건물 형태
    "FA-06": "building_5",      # 미확정: 일반 건물 형태(무기고 특유의 형태는 안 보임)
}
# 시설물 상태 코드(JSON 출력값) - 전부 영문 코드로 통일
FACILITY_STATUS_OPTIONS = ["normal", "destroy", "fire", "unconfirmed"]


# ---------------------------------------------------------
# 5. 폭파구 실측 치수표 (단위: mm, 이미지 16 기준)
#    (가로, 세로, 높이/깊이) - 키 값(big/medium/small)이 곧 JSON 출력 코드
# ---------------------------------------------------------
CRATER_SIZE_TABLE_MM = {
    "big": {"w": 179.0, "h": 200.0, "d": 30.0},      # 대형
    "medium": {"w": 159.0, "h": 150.0, "d": 23.0},   # 중형
    "small": {"w": 102.5, "h": 99.0, "d": 16.0},     # 소형
}

# ---------------------------------------------------------
# 6. 불발탄 실측 치수표 (단위: mm, 이미지 8 기준)
#    키 값(cluster/dumb/missile)이 곧 JSON 출력 코드
# ---------------------------------------------------------
UXO_SIZE_TABLE_MM = {
    "cluster": {"w": 28.0, "h": 28.0, "d": 20.5},   # 자탄(집속탄 내부), 구형에 가까움
    "dumb": {"w": 44.0, "h": 44.0, "d": 93.0},      # 포탄(일반 투하/포병탄), 원통형
    "missile": {"w": 50.0, "h": 50.0, "d": 115.0},  # 미사일, 가장 길쭉함
}

# ---------------------------------------------------------
# 7. 미션 코드 (mission_code)  -  실전 시작 전 갱신 필요
# ---------------------------------------------------------
MISSION_CODE = "SHUUFT9A"  # 실제 대회에서 팀별로 부여된 코드

# ---------------------------------------------------------
# 8. 임무 제한시간
# ---------------------------------------------------------
MISSION_TIME_LIMIT_SEC = 180

# ---------------------------------------------------------
# 9. 탐지 백엔드 설정 (고전 CV <-> YOLO11n 전환)
#    확정된 파이프라인은 YOLO11n 배치 추론이 기본이지만, 가중치가 아직 없는 개발/합성테스트
#    단계에서는 "classical"로 바로 검증할 수 있게 토글을 유지합니다.
# ---------------------------------------------------------
# 대회 현장에서 테스트 기간 중 촬영한 이미지로 YOLO11n 학습이 끝나면
# 아래 두 값만 "yolo"로 바꾸면 src/pipeline.py 수정 없이 백엔드가 전환됩니다.
# (src/detection.py의 build_object_detector()/build_facility_classifier() 참고)
DETECTOR_BACKEND = "yolo"         # "classical" | "yolo" - 폭파구/불발탄 통합 탐지 (확정 파이프라인: yolo)
FACILITY_BACKEND = "hybrid"        # "classical" | "yolo" | "hybrid" - 시설물 상태(normal/destroy/fire) 분류
# hybrid: fire 판정만 고전 CV(detection.classify_fire_by_contrast, precision 100%로 검증됨)가 전담하고
# fire 아니면 YOLO(destroy/normal 판정)에 위임. destroy의 붉은 파손 스티커를 fire로 오인하는 문제를
# 근본적으로 차단하고 싶으면 "hybrid"로 바꿀 것 (2026-07-15 yolo_facility1+2 552장으로 검증).

# --- 폭파구/불발탄 YOLO 모델 설정 (3-A: zone 타일 배치 추론) ---
# 2026-07-14: 3rdtry.yolov11 + yolo_obj_dataset(실사진) + enhancement/generated_dataset(합성 100장)
# 병합(총 114장, 88%가 합성)으로 학습한 yolo11n_object.pt는 val mAP50 0.98로 훈련 지표는 좋았지만
# val이 대부분 합성 이미지라 실제 드론 영상에서는 활주로(RW) 탐지를 거의 놓치고 헛탐지도 발생함
# (debug_visualize.py로 확인). 그래서 이전에 실사진 위주로 학습된 yolov8n_object_v2.pt로 되돌림 -
# 이쪽이 실제 영상 기준으로 더 정확함이 확인됨. yolo11n_object.pt는 재학습 전까지 보류.
# 클래스 순서는 알파벳순(0:big,1:cluster,2:dumb,3:medium,4:missile,5:small)을 그대로 따름.
# 나중에 다른 학습 데이터셋을 쓸 경우 반드시 그 data.yaml의 names 순서와 다시 맞출 것.
YOLO_OBJECT_WEIGHTS = "models/yolo11n_object_final.pt"
YOLO_OBJECT_CONF_THRESHOLD = 0.4
# 학습 클래스 idx -> (category, subtype[영문 코드=JSON 출력값]). data.yaml의 names 순서와 반드시 일치시킬 것.
YOLO_OBJECT_CLASS_MAP = {
    0: ("crater", "big"),
    1: ("uxo", "cluster"),
    2: ("uxo", "dumb"),
    3: ("crater", "medium"),
    4: ("uxo", "missile"),
    5: ("crater", "small"),
}

# --- 시설물 상태 YOLO11n-cls 모델 설정 (3-B: 고정 6좌표 배치 추론) ---
YOLO_FACILITY_WEIGHTS = "models/yolo11n_facility_cls.pt"
YOLO_FACILITY_CONF_THRESHOLD = 0.4
# 학습 클래스 idx -> 상태 코드(영문=JSON 출력값).
# ultralytics 분류(cls) 학습은 클래스 폴더명을 '알파벳순'으로 정렬해 인덱스를 부여함
#   (normal/destroy/fire 폴더 -> 0:destroy, 1:fire, 2:normal). 아래 맵도 그 순서에 맞춰 둠.
# 단, 추론 시 detection.YoloFacilityClassifier가 학습된 모델의 result.names(폴더명 그대로)를
# 우선 사용하므로, 분류 폴더명을 normal/destroy/fire로만 두면 정렬 순서가 바뀌어도 자동으로 맞음.
# 이 맵은 model.names를 못 얻는 예외 상황의 폴백일 뿐임.
YOLO_FACILITY_CLASS_MAP = {
    0: "destroy",
    1: "fire",
    2: "normal",
}


# ---------------------------------------------------------
# 10. 탑뷰(bird's eye) 워핑 설정
#     ArUco 호모그래피 계산 후, 프레임 전체를 실좌표계와 픽셀이 1:1 정렬되는
#     탑뷰 이미지로 워핑합니다(calibration.warp_to_topview). 이후 모든 crop/타일 분할은
#     이 탑뷰 이미지의 픽셀 좌표만으로 계산 가능합니다(프레임별 역투영 불필요).
# ---------------------------------------------------------
WARP_PX_PER_CM = 4  # 탑뷰 워핑 해상도(모형 1cm당 픽셀 수). 실제 카메라 해상도에 맞춰 조정 필요.
WARP_CANVAS_WIDTH_PX = int(FIELD_WIDTH_CM * WARP_PX_PER_CM)
WARP_CANVAS_HEIGHT_PX = int(FIELD_HEIGHT_CM * WARP_PX_PER_CM)


# ---------------------------------------------------------
# 11. zone 타일 분할 순서 (3-A 배치 추론 대상, 시설물 제외)
#     탑뷰 이미지에서 zone별로 경계 걸침 없이 그리드 crop -> 한 번의 batch predict로 처리.
# ---------------------------------------------------------
ZONE_TILE_ORDER = RUNWAY_SEGMENT_ORDER + TAXIWAY_A_ORDER + TAXIWAY_B_ORDER  # 현재 20개
# "한 번에 30개 타일 배치 추론"은 배치 크기 상한을 의미(zone 개수가 늘어나도 코드 변경 없이 대응).
YOLO_BATCH_SIZE_MAX = 30


# ---------------------------------------------------------
# 12. 영상 입력 파이프라인 (watchdog 감시 + 프레임 추출)
# ---------------------------------------------------------
VIDEO_INPUT_DIR = r"C:\Users\Admin\Documents\filerecvsender\local_inbox"  # 이 폴더에 영상 파일이 생성되면 감지
VIDEO_FILE_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
FRAME_EXTRACT_INTERVAL = 30                 # N프레임마다 1장 추출 (실측 드론 영상 fps 확인 후 갱신 필요)
VIDEO_STABLE_WAIT_SEC = 1.0                 # 파일 크기가 더 이상 변하지 않을 때까지 대기(쓰기 완료 판단)
FRAME_OUTPUT_DIR = "test_images"            # 추출된 프레임이 저장되는 폴더(파이프라인 입력 폴더와 동일)


# ---------------------------------------------------------
# 13. 대시보드 전송 설정 (6단계) - 실제 엔드포인트는 대회 현장에서 확정
# ---------------------------------------------------------
DASHBOARD_ENDPOINT_URL = "http://192.168.250.2:8080"  # 대회 측 전송 주소 (실제 요청 시 뒤에 /{MISSION_CODE}가 자동으로 붙음)
# 8개 JSON 전송 순서 (schemas.py의 파일명 <-> outputs 딕셔너리 키와 동일)
TRANSMIT_ORDER = [
    "start", "crater_detect", "crater_count", "runway_status",
    "facility_status", "uxo_detect", "uxo_count", "report",
]
TRANSMIT_DUPLICATE_COUNT = 1   # 수신 유실 대비, 동일 JSON을 순차로 몇 번 중복 전송할지
TRANSMIT_TIMEOUT_SEC = 5

# 2026-07-14 현장 테스트로 확인: uxo_detect 보고는 서버가 최대 이 개수(구간)까지만 받고,
# 초과하면 보고 전체를 HTTP 400으로 거부함("uxo_detect 보고는 6구간 이하로만 전송할 수
# 있습니다"). 초과분은 신뢰도(confidence) 낮은 순으로 잘라서 보냄(pipeline.py 참고).
UXO_DETECT_MAX_ENTRIES = 6

# crater_detect도 동일한 방식으로 서버가 최대 이 개수까지만 받음(초과 시 보고 전체 거부).
CRATER_DETECT_MAX_ENTRIES = 5

# crater_detect/uxo_detect 둘 다 빈 배열(0건)이면 서버가 거부함 - 탐지가 하나도 없어도
# 최소 이 개수만큼은 채워서 보내야 함(pipeline.py의 _enforce_count_bounds 참고).
CRATER_DETECT_MIN_ENTRIES = 1
UXO_DETECT_MIN_ENTRIES = 1


# ---------------------------------------------------------
# 14. LLM 상황보고서 글자수 제약 (5단계) - 대회 규정: 공백 포함 100자 이내
# ---------------------------------------------------------
REPORT_MIN_CHARS = 50
REPORT_MAX_CHARS = 100

# ---------------------------------------------------------
# 15. 활주로 상태/운용여부 판정 기준 (상황보고서용, 대회 규정)
#     가용길이(m) 구간별로 상태와 운용여부를 판정.
# ---------------------------------------------------------
RUNWAY_STATUS_THRESHOLDS = [
    # (최소 가용길이(m, 이상), 상태, 운용여부)
    (2100, "정상", "사용 가능"),
    (1500, "제한 운용", "제한적 사용 가능"),
    (900, "비상 운용", "사용 가능 여부 검토"),
    (0, "운용 불가", "사용 불가(폐쇄)"),
]


def classify_runway_status(available_length_m: float):
    """활주로 가용길이(m)에 따른 (상태, 운용여부) 판정."""
    for threshold, status, availability in RUNWAY_STATUS_THRESHOLDS:
        if available_length_m >= threshold:
            return status, availability
    return RUNWAY_STATUS_THRESHOLDS[-1][1], RUNWAY_STATUS_THRESHOLDS[-1][2]
