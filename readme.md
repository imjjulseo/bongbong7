# 공군 해커톤 AI경진대회 - 정찰/전투피해평가(BDA) 파이프라인
=======

드론이 촬영한 디오라마 영상에서 **폭파구/시설물 피해/불발탄**을 탐지하고,
대회 규격(mission_code 표)에 맞는 8개 JSON 파일을 자동 생성하는 전체 파이프라인입니다.

## 왜 이렇게 설계했는가 (핵심 차별화 포인트)

1. **좌표 통일**: ArUco 마커로 픽셀↔실좌표(cm) 변환행렬(호모그래피)을 계산해,
   모든 결과(폭파구 위치, 활주로 가용길이 등)를 실제 미터 단위로 정확히 산출합니다.
2. **중복 제거**: 드론이 같은 지점을 여러 프레임 촬영해도, 실좌표 기준으로 가까운
   탐지끼리 묶어(geo-referenced dedup) 개수를 부풀리지 않습니다.
3. **폭파구 vs 불발탄 통합 분류**: 두 물체 모두 "어두운 blob"이라는 공통점이 있어
   따로 탐지하면 이중 카운트가 발생합니다. 하나의 탐지 결과를 실측 치수표와
   대조해 사후적으로 분류하므로 중복이 없습니다.
4. **시설물 6슬롯 강제 매핑**: 시설물 위치가 고정되어 있다는 사전 지식을 이용해,
   탐지에 실패해도 슬롯 자체는 항상 6개가 채워집니다('미확인'으로 표시) →
   "시설물 누락" 감점을 원천 차단합니다.
5. **로컬 LLM + 오프라인 폴백**: 클라우드 API 키가 없으므로 로컬 LLM(Ollama 등)을
   우선 시도하되, 실패하면 결정론적 템플릿으로 자동 전환되어 임무가 절대
   중단되지 않습니다.
6. **자동 검증(QA)**: 전송 전 JSON 정합성(개수 일치, 슬롯 누락 등)을 자동 점검합니다.
7. **고전 CV ↔ YOLO 백엔드 전환**: 탐지/분류 로직을 공통 인터페이스로 감싸,
   대회 현장에서 테스트 기간 중 촬영한 이미지로 YOLO 학습이 끝나면
   `field_config.py`의 설정값만 바꿔서 코드 수정 없이 전환할 수 있습니다
   (자세한 내용은 아래 "탐지 백엔드 전환" 참고).

## 폴더 구조

```
airbase_bda/
├── config/
│   └── field_config.py      # 경기장 레이아웃, 실측 치수표, 축척 등 모든 고정값
├── src/
│   ├── schemas.py            # 8개 미션 JSON 스키마
│   ├── calibration.py        # ArUco 캘리브레이션 (픽셀<->실좌표 변환)
│   ├── detection.py           # 폭파구/불발탄 통합 탐지 + 시설물 상태 분류
│   │                           # (고전 CV / YOLO 백엔드를 공통 인터페이스로 감쌈)
│   ├── geo_dedup.py           # 실좌표 기반 중복 제거
│   ├── runway_analysis.py     # 활주로 최장 가용구간 알고리즘
│   ├── facility_analysis.py   # 시설물 6슬롯 강제 매핑
│   ├── uxo_analysis.py        # 구간 배정 및 활주로 내 개수 집계
│   ├── report_generator.py    # 로컬 LLM 보고서 생성 + 오프라인 폴백
│   ├── validator.py           # 전송 전 자동 검증(QA)
│   └── pipeline.py            # 전체 흐름 오케스트레이션
├── scripts/
│   ├── generate_test_scene.py # 합성 테스트 이미지 생성 (실제 드론 사진 없이 검증용)
│   ├── run_mission.py         # 메인 실행 진입점
│   └── train_yolo.py          # 테스트 기간 촬영 이미지로 YOLO 학습(전환용)
├── tests/
│   └── test_core.py           # 핵심 로직 단위 테스트 (12개, 전부 통과 확인됨)
└── output/                    # 실행 결과 JSON이 저장되는 폴더
```

## 사용법

### 1) 설치
```bash
pip install -r requirements.txt
```

### 2) 합성 테스트로 전체 파이프라인 검증 (실제 드론 사진 없이)
```bash
python scripts/run_mission.py --synthetic
```
자동으로 ArUco 마커 + 폭파구 + 불발탄 + 시설물 피해가 배치된 테스트 이미지를 만들고,
전체 파이프라인(캘리브레이션→탐지→분석→보고서)을 실행한 뒤 결과를 요약 출력합니다.

### 3) 실제 촬영 이미지로 실행
```bash
python scripts/run_mission.py --images /path/to/frames_directory
```

### 4) 로컬 LLM 없이 템플릿 보고서만 사용
```bash
python scripts/run_mission.py --synthetic --no-llm
```

### 5) 단위 테스트 실행
```bash
python tests/test_core.py
# 또는: python -m pytest tests/ -v
```

## 탐지 백엔드 전환 (고전 CV ↔ YOLO)

`src/detection.py`는 폭파구/불발탄 탐지와 시설물 상태 분류를 각각
`ObjectDetectorBackend` / `FacilityStatusBackend` 공통 인터페이스로 감싸두었습니다.
지금은 둘 다 고전 CV(`ClassicalBlobDetector`, `ClassicalFacilityClassifier`)로 동작하며,
대회 현장에서 테스트 기간 중 촬영한 이미지로 YOLO 학습이 끝나면 아래처럼 전환합니다.

1. `scripts/train_yolo.py`로 학습 (자세한 사용법은 스크립트 상단 docstring 참고):
   ```bash
   python scripts/train_yolo.py --task detect --data data/object.yaml \
       --model yolov8n.pt --epochs 100 --out models/object_detector
   python scripts/train_yolo.py --task classify --data data/facility_cls \
       --model yolov8n-cls.pt --epochs 50 --out models/facility_classifier
   ```
2. `config/field_config.py`에서 가중치 경로와 백엔드를 갱신:
   ```python
   DETECTOR_BACKEND = "yolo"
   YOLO_OBJECT_WEIGHTS = "models/object_detector.pt"
   FACILITY_BACKEND = "yolo"
   YOLO_FACILITY_WEIGHTS = "models/facility_classifier.pt"
   ```
   학습 클래스 순서는 `YOLO_OBJECT_CLASS_MAP` / `YOLO_FACILITY_CLASS_MAP`과
   반드시 일치해야 합니다(= data.yaml의 `names` 순서).
3. `src/pipeline.py`는 수정할 필요 없이 그대로 `python scripts/run_mission.py ...`를 실행하면 됩니다.
   특정 실행만 다른 백엔드로 돌려보고 싶다면
   `python scripts/run_mission.py --images ... --detector-backend yolo` 처럼
   CLI 인자로도 override 할 수 있습니다.
4. 고전 CV로 즉시 되돌리려면 `DETECTOR_BACKEND`/`FACILITY_BACKEND`를 다시
   `"classical"`로 바꾸면 됩니다 — 대회 당일 YOLO 추론이 불안정할 때의 안전장치입니다.

`ultralytics`는 `"yolo"` 백엔드를 실제로 쓸 때만 필요합니다
(`requirements.txt`의 주석 처리된 줄을 해제하고 설치).

## 대회 전 반드시 해야 할 일 (체크리스트)

- [ ] `config/field_config.py`의 `ARUCO_MARKER_WORLD_POSITIONS`을 **테스트 경기장에서
      실측한 실제 마커 좌표**로 갱신
- [ ] `FACILITY_TYPE_BY_SLOT`(FA-01~06과 시설물 종류 매핑)을 실제 배치와 대조해 확인
- [ ] `MISSION_CODE`를 대회에서 부여받은 실제 8자리 코드로 교체
- [ ] 로컬 LLM(Ollama 등) 사용 가능 여부와 사양을 **운영진에게 사전 확인** —
      "생성형 AI API 키 미제공"이 클라우드 금지와 결합될 때 로컬 LLM이
      규정 위반이 아닌지 반드시 확인 필요
- [ ] 대회 PC 사양에서 로컬 LLM 모델(`report_generator.py`의 `OLLAMA_MODEL`) 속도 벤치마크
- [ ] 실제 드론 촬영 영상으로 `dark_threshold`, `min_area_px` 등 탐지 파라미터 재조정
      (`detection.py`의 `detect_dark_blobs` 인자, 고전 CV 백엔드 기준)
- [ ] 테스트 기간 중 촬영한 이미지가 쌓이면 `scripts/train_yolo.py`로 YOLO 학습 후
      "탐지 백엔드 전환" 절차대로 `DETECTOR_BACKEND`/`FACILITY_BACKEND`를 `"yolo"`로 전환
      (합성 데이터로 사전학습 후 전이학습 권장)

## 알려진 한계 (합성 테스트 기준)

- 고전 CV(색상/형태 기반) 탐지는 매우 작은 물체(예: 자탄 크기 실측 28mm급)는
  해상도에 따라 놓칠 수 있습니다. 실제 드론 촬영 해상도에 맞춰
  `min_area_px` 값을 낮추거나 SAHI(타일 분할 추론) 적용을 권장합니다.
  (YOLO 백엔드로 전환하면 학습 데이터에 작은 객체가 충분히 포함될 경우 개선될 수 있음)
- 화재/파손 판정은 색상 휴리스틱 기반이라 조명 조건에 민감할 수 있습니다.
  실전 연습에서 임계값(`ClassicalFacilityClassifier`)을 재조정하거나 YOLO 백엔드로 전환하세요.
