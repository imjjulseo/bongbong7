# 공군 해커톤 AI경진대회 - 정찰/전투피해평가(BDA) 파이프라인


전송 주소: http://192.168.250.2:8080/SHUUFT9A

드론이 촬영한 디오라마 **영상**에서 **폭파구/시설물 피해/불발탄**을 탐지하고,
대회 규격(mission_code 표)에 맞는 8개 JSON 파일을 자동 생성해 대시보드로 전송하는
전체 파이프라인입니다. 아래는 **확정된 파이프라인**을 그대로 구현한 것입니다.

## 확정 파이프라인 (6단계)

1. **영상 입력**: 지정 폴더(`video_input/`)에 영상 파일이 들어오면 `watchdog`이 감지 →
   지정 프레임 간격(`FRAME_EXTRACT_INTERVAL`)마다 이미지로 추출 (`scripts/video_watcher.py`)
2. **탑뷰 워핑**: 각 프레임에서 ArUco 마커 검출 → 호모그래피 계산 → 프레임 전체를
   경기장 실좌표계와 픽셀이 1:1 정렬되는 탑뷰(bird's eye) 이미지로 워핑
   (`calibration.warp_to_topview`). 이 단계 이후로는 픽셀좌표 == 실좌표(cm) * px_per_cm이므로
   프레임별 역투영 없이 단순 슬라이싱만으로 crop 가능합니다.
3. **3-A [활주로/유도로]**: 워핑된 이미지를 zone(RW-01~10, TW-A1~5, TW-B1~5, 총 20개)별
   타일로 crop(경계 걸침 없는 단순 그리드 → overlap 불필요) → YOLO11n 배치 추론
   (한 번의 `model.predict(list)` 호출로 처리) → 폭파구 3클래스(`big/medium/small`) +
   불발탄 3클래스(`missile/dumb/cluster`) 탐지 → zone별 결과 취합 → 가용거리 계산
   (`runway_analysis`, 로직 확정)
4. **3-B [시설물]**: 고정 좌표 6개(FA-01~06) crop → YOLO11n-cls 배치 추론
   (마찬가지로 한 번의 `predict(list)` 호출) → `normal/destroy/fire` 3클래스 분류
5. **결과 통합**: 3-A, 3-B 결과를 mission_code 포함 8종 JSON 필드로 매핑
   (코드값은 전부 영문: `big/medium/small`, `missile/dumb/cluster`, `normal/destroy/fire/unconfirmed`)
6. **상황보고서**: JSON 취합 → LLM 프롬프트(50~100자 제약 명시) → `report.json`
7. **전송**: 전송 모듈로 대시보드에 순차/중복 전송 (`src/transmitter.py`, 현재는 엔드포인트
   미정으로 스텁 상태 — `DASHBOARD_ENDPOINT_URL` 설정 시 자동으로 실제 전송 전환)

## 왜 이렇게 설계했는가 (핵심 차별화 포인트)

1. **좌표 통일 + 탑뷰 워핑**: ArUco 마커로 호모그래피를 계산한 뒤 프레임 자체를
   실좌표계와 정렬된 탑뷰 이미지로 워핑합니다. 이후 zone 타일/시설물 ROI crop이
   프레임별 역투영 없이 단순 픽셀 슬라이싱만으로 끝나 파이프라인이 단순해집니다.
2. **경계 걸침 없는 그리드 분할**: zone끼리 겹치지 않는 좌표 정의를 그대로 crop
   경계로 사용하므로 overlap 처리(SAHI 등)가 필요 없습니다.
3. **배치 추론**: zone 타일(3-A)과 시설물 ROI(3-B) 모두 한 번의 `model.predict(list)`
   호출로 배치 처리합니다(`detection.detect_zone_tiles` / `classify_frame_batch`).
4. **중복 제거**: 드론이 같은 지점을 여러 프레임 촬영해도, 실좌표 기준으로 가까운
   탐지끼리 묶어(geo-referenced dedup) 개수를 부풀리지 않습니다.
5. **시설물 6슬롯 강제 매핑**: 시설물 위치가 고정되어 있다는 사전 지식을 이용해,
   탐지에 실패해도 슬롯 자체는 항상 6개가 채워집니다('unconfirmed'로 표시) →
   "시설물 누락" 감점을 원천 차단합니다.
6. **로컬 LLM + 오프라인 폴백**: 클라우드 API 키가 없으므로 로컬 LLM(Ollama 등)을
   우선 시도하되, 실패하면 결정론적 템플릿으로 자동 전환되어 임무가 절대
   중단되지 않습니다. 두 경로 모두 50~100자 글자수 제약을 강제 적용합니다.
7. **자동 검증(QA)**: 전송 전 JSON 정합성(개수 일치, 슬롯 누락, 글자수 제약 등)을
   자동 점검합니다.
8. **고전 CV ↔ YOLO11n 백엔드 전환**: 탐지/분류 로직을 공통 인터페이스로 감싸,
   YOLO11n 가중치가 준비되기 전에도 `field_config.py`의 설정값만 바꿔서
   고전 CV로 전체 파이프라인을 바로 검증할 수 있습니다
   (자세한 내용은 아래 "탐지 백엔드 전환" 참고).
9. **전송 스텁**: 대시보드 API 엔드포인트가 대회 현장에서 확정되기 전까지는
   파일 저장은 100% 보장하고, 전송은 로그만 남기는 안전한 스텁으로 동작합니다.

## 폴더 구조

```
airbase_bda/
├── config/
│   └── field_config.py      # 경기장 레이아웃, 실측 치수표, 워핑/영상/전송/리포트 설정 등 모든 고정값
├── src/
│   ├── schemas.py            # 8개 미션 JSON 스키마
│   ├── calibration.py        # ArUco 캘리브레이션 + 탑뷰(bird's eye) 워핑
│   ├── tiling.py              # 탑뷰 이미지 -> zone 타일(3-A) / 시설물 ROI(3-B) crop
│   ├── detection.py           # 폭파구/불발탄 배치 탐지 + 시설물 상태 배치 분류
│   │                           # (고전 CV / YOLO11n 백엔드를 공통 인터페이스로 감쌈)
│   ├── geo_dedup.py           # 실좌표 기반 중복 제거
│   ├── runway_analysis.py     # 활주로 최장 가용구간 알고리즘 (확정 로직)
│   ├── facility_analysis.py   # 시설물 6슬롯 강제 매핑
│   ├── uxo_analysis.py        # 활주로 구간 내 개수 집계
│   ├── report_generator.py    # 로컬 LLM 보고서 생성(50~100자) + 오프라인 폴백
│   ├── validator.py           # 전송 전 자동 검증(QA)
│   ├── transmitter.py         # 대시보드 순차/중복 전송 (엔드포인트 확정 전까지 스텁)
│   └── pipeline.py            # 전체 흐름 오케스트레이션 (2~6단계)
├── scripts/
│   ├── video_watcher.py       # 1단계: watchdog 영상 감시 + 프레임 추출 + 파이프라인 자동 실행
│   ├── generate_test_scene.py # 합성 테스트 이미지 생성 (실제 드론 사진 없이 검증용)
│   ├── run_mission.py         # 프레임 폴더/합성 이미지로 파이프라인을 1회 실행하는 진입점
│   └── train_yolo.py          # 테스트 기간 촬영 이미지로 YOLO11n 학습(전환용)
├── tests/
│   └── test_core.py           # 핵심 로직 단위 테스트
└── output/                    # 실행 결과 JSON이 저장되는 폴더
```

## 사용법

### 1) 설치
```bash
pip install -r requirements.txt
```

### 2) 합성 테스트로 전체 파이프라인 검증 (실제 드론 사진/영상 없이)
```bash
python scripts/run_mission.py --synthetic
```
자동으로 ArUco 마커 + 폭파구 + 불발탄 + 시설물 피해가 배치된 테스트 이미지를 만들고,
전체 파이프라인(워핑→타일 배치추론→분석→보고서)을 실행한 뒤 결과를 요약 출력합니다.

### 3) 실제 촬영 이미지(프레임)로 1회 실행
```bash
python scripts/run_mission.py --images /path/to/frames_directory
```

### 4) 영상 입력부터 자동으로 (1단계 watchdog 포함)
```bash
# 폴더에 이미 있는 영상들만 한 번 처리
python scripts/video_watcher.py --once

# 폴더를 계속 감시하며 새 영상이 들어올 때마다 자동 처리
python scripts/video_watcher.py --watch
```
`config/field_config.py`의 `VIDEO_INPUT_DIR`(기본 `video_input/`)에 영상 파일을 넣으면
`FRAME_EXTRACT_INTERVAL` 프레임마다 이미지를 추출하고, 바로 이어서 파이프라인을 실행합니다.

### 5) 로컬 LLM 없이 템플릿 보고서만 사용 / 대시보드 전송까지 실행
```bash
python scripts/run_mission.py --synthetic --no-llm
python scripts/run_mission.py --synthetic --send   # 6단계(전송)까지 실행
```

### 6) 단위 테스트 실행
```bash
python tests/test_core.py
# 또는: python -m pytest tests/ -v
```

## 탐지 백엔드 전환 (고전 CV ↔ YOLO11n)

`src/detection.py`는 폭파구/불발탄 탐지와 시설물 상태 분류를 각각
`ObjectDetectorBackend` / `FacilityStatusBackend` 공통 인터페이스로 감싸두었습니다.
지금은 둘 다 고전 CV(`ClassicalBlobDetector`, `ClassicalFacilityClassifier`)로 동작하며,
**확정 파이프라인의 기본 백엔드는 YOLO11n**입니다 — 대회 현장에서 테스트 기간 중 촬영한
이미지로 학습이 끝나면 아래처럼 전환합니다.

1. `scripts/train_yolo.py`로 학습 (자세한 사용법은 스크립트 상단 docstring 참고):
   ```bash
   python scripts/train_yolo.py --task detect --data data/object.yaml \
       --model yolo11n.pt --epochs 100 --out models/yolo11n_object
   python scripts/train_yolo.py --task classify --data data/facility_cls \
       --model yolo11n-cls.pt --epochs 50 --out models/yolo11n_facility_cls
   ```
   detect 학습 클래스 순서: `big, medium, small, missile, dumb, cluster`
   (`field_config.YOLO_OBJECT_CLASS_MAP`과 반드시 일치)
   classify 학습 폴더 구성: `normal/ destroy/ fire/` (`field_config.YOLO_FACILITY_CLASS_MAP`과 일치)
2. `config/field_config.py`에서 가중치 경로와 백엔드를 갱신:
   ```python
   DETECTOR_BACKEND = "yolo"
   YOLO_OBJECT_WEIGHTS = "models/yolo11n_object.pt"
   FACILITY_BACKEND = "yolo"
   YOLO_FACILITY_WEIGHTS = "models/yolo11n_facility_cls.pt"
   ```
3. `src/pipeline.py`는 수정할 필요 없이 그대로 실행하면 됩니다.
   특정 실행만 다른 백엔드로 돌려보고 싶다면
   `python scripts/run_mission.py --images ... --detector-backend yolo` 처럼
   CLI 인자로도 override 할 수 있습니다.
4. 고전 CV로 즉시 되돌리려면 `DETECTOR_BACKEND`/`FACILITY_BACKEND`를 다시
   `"classical"`로 바꾸면 됩니다 — YOLO11n 추론이 불안정할 때의 안전장치입니다.

`ultralytics`는 `"yolo"` 백엔드를 실제로 쓸 때만 필요합니다.

## 대회 전 반드시 해야 할 일 (체크리스트)

- [ ] `config/field_config.py`의 `ARUCO_MARKER_WORLD_POSITIONS`을 **테스트 경기장에서
      실측한 실제 마커 좌표**로 갱신
- [ ] `FACILITY_TYPE_BY_SLOT`(FA-01~06과 시설물 종류 매핑, 현재 영문 코드는 placeholder)를
      원본 슬라이드 확정치로 교체
- [ ] `MISSION_CODE`를 대회에서 부여받은 실제 8자리 코드로 교체
- [ ] `DASHBOARD_ENDPOINT_URL`(전송 API), `TRANSMIT_ORDER`/`TRANSMIT_DUPLICATE_COUNT`를
      대회 대시보드 스펙에 맞춰 확정 (현재는 스텁)
- [ ] `WARP_PX_PER_CM`(탑뷰 워핑 해상도), `FRAME_EXTRACT_INTERVAL`(영상 프레임 추출 간격)을
      실제 드론 영상 해상도/fps에 맞춰 재조정
- [ ] 로컬 LLM(Ollama 등) 사용 가능 여부와 사양을 **운영진에게 사전 확인** —
      "생성형 AI API 키 미제공"이 클라우드 금지와 결합될 때 로컬 LLM이
      규정 위반이 아닌지 반드시 확인 필요
- [ ] 대회 PC 사양에서 로컬 LLM 모델(`report_generator.py`의 `OLLAMA_MODEL`) 속도 벤치마크
- [ ] 실제 드론 촬영 영상으로 `dark_threshold`, `min_area_px` 등 탐지 파라미터 재조정
      (`detection.py`의 `detect_dark_blobs` 인자, 고전 CV 백엔드 기준)
- [ ] 테스트 기간 중 촬영한 이미지가 쌓이면 `scripts/train_yolo.py`로 YOLO11n 학습 후
      "탐지 백엔드 전환" 절차대로 `DETECTOR_BACKEND`/`FACILITY_BACKEND`를 `"yolo"`로 전환

## 알려진 한계 (합성 테스트 기준)

- zone 경계에 걸친 물체는 grid 분할 특성상 두 타일로 쪼개져 잡힐 수 있습니다
  (overlap 없는 단순 그리드 분할을 확정 스펙으로 채택했기 때문).
- 고전 CV(색상/형태 기반) 탐지는 매우 작은 물체(예: cluster 실측 28mm급)는
  해상도에 따라 놓칠 수 있습니다. `WARP_PX_PER_CM`을 높이거나 YOLO11n 백엔드로
  전환하면 개선될 수 있습니다.
- 화재/파손 판정(고전 CV 백엔드)은 색상 휴리스틱 기반이라 조명 조건에 민감할 수 있습니다.
  실전 연습에서 임계값(`ClassicalFacilityClassifier`)을 재조정하거나 YOLO11n 백엔드로 전환하세요.
- `src/transmitter.py`는 `DASHBOARD_ENDPOINT_URL`이 설정되기 전까지 실제 네트워크 전송을
  하지 않는 스텁입니다.
