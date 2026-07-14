import cv2
import os
import shutil
import random
import yaml
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 및 클래스 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "cropped_facilities"
DATASET_DIR = BASE_DIR / "yolo_facility_dataset"

# 3가지 클래스 정의 (키보드 1, 2, 3 입력과 매핑)
CLASS_MAP = {
    ord('1'): "normal",
    ord('2'): "destroy",
    ord('3'): "fire"
}

# 학습(train)과 검증(val) 데이터 분할 비율 (예: 8:2)
TRAIN_RATIO = 1

# ---------------------------------------------------------
# 2. YOLO Classification 폴더 구조 생성
# ---------------------------------------------------------
def create_dataset_structure():
    if DATASET_DIR.exists():
        print(f"[주의] '{DATASET_DIR.name}' 폴더가 이미 존재합니다. 이어서 작업합니다.")
    
    for split in ["train", "val"]:
        for class_name in CLASS_MAP.values():
            (DATASET_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
            
def generate_yaml():
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_data = {
        "path": str(DATASET_DIR.absolute()),
        "train": "train",
        "val": "val",
        "nc": len(CLASS_MAP),
        "names": list(CLASS_MAP.values())
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[+] '{yaml_path.name}' 파일이 생성되었습니다.")

# ---------------------------------------------------------
# 3. 메인 라벨링 로직
# ---------------------------------------------------------
def fast_keyboard_labeling():
    create_dataset_structure()
    generate_yaml()

    # 모든 하위 폴더에서 png 파일 수집
    # 예: cropped_facilities/0/FA-01.png
    all_images = list(INPUT_DIR.rglob("*.png"))
    
    if not all_images:
        print(f"'{INPUT_DIR}' 폴더에 이미지가 없습니다.")
        return

    print("\n" + "="*50)
    print("🚀 초고속 키보드 라벨링을 시작합니다!")
    print("  [1] Normal  (정상)")
    print("  [2] Destroy (손상)")
    print("  [3] Fire    (화재)")
    print("  [Space] 건너뛰기")
    print("  [Q 또는 ESC] 저장하고 종료")
    print("="*50 + "\n")

    labeled_count = 0
    skipped_count = 0
    total = len(all_images)

    for i, img_path in enumerate(all_images):
        # 1. 이미지 로드 및 리사이즈 (보기 편하게 2배 확대)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        display_img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # 2. 이미지에 안내 텍스트 표시
        # 원본 폴더 번호와 파일명 추출 (예: "0", "FA-01")
        parent_folder = img_path.parent.name
        img_name = img_path.name
        
        info_text = f"[{i+1}/{total}] {parent_folder}/{img_name}"
        cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img, "1:Normal | 2:Destroy | 3:Fire | Q:Quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 3. 이미지 출력 및 키 입력 대기
        cv2.imshow("Fast Labeling", display_img)
        key = cv2.waitKey(0) & 0xFF

        # 4. 키 입력에 따른 처리
        if key in [ord('q'), ord('Q'), 27]: # 27은 ESC
            print("\n작업을 중단하고 종료합니다.")
            break
            
        elif key in CLASS_MAP:
            class_name = CLASS_MAP[key]
            
            # Train / Val 랜덤 분배
            split = "train" if random.random() < TRAIN_RATIO else "val"
            
            # 파일 이름 충돌 방지 (폴더명_파일명 형식으로 저장)
            # 예: "0_FA-01.png"
            new_filename = f"{parent_folder}_{img_name}"
            dest_path = DATASET_DIR / split / class_name / new_filename
            
            # 이미지 복사 (원본 보존)
            shutil.copy(img_path, dest_path)
            labeled_count += 1
            print(f"  -> [{class_name.upper()}] 분류됨 (저장: {split}/{class_name}/{new_filename})")
            
        elif key == 32: # Spacebar
            skipped_count += 1
            print(f"  -> 건너뜀")
        else:
            # 잘못된 키 입력 시 현재 이미지를 다시 띄우기 위해 인덱스 보정은 생략하고 건너뜁니다.
            # (엄밀하게는 i를 줄여야 하지만 편리상 스킵 처리)
            skipped_count += 1
            print(f"  -> 잘못된 입력 (건너뜀)")

    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print(f"✅ 라벨링 작업 완료!")
    print(f"  - 총 이미지: {total}장")
    print(f"  - 라벨링 완료: {labeled_count}장")
    print(f"  - 건너뜀: {skipped_count}장")
    print(f"  - 저장 위치: {DATASET_DIR.absolute()}")
    print("="*50)

if __name__ == "__main__":
    fast_keyboard_labeling()