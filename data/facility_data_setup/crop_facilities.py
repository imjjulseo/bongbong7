import cv2
import sys
import os
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 및 모듈 설정
# ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc

BASE_DIR = Path(__file__).resolve().parent

PADDING_LENGTH = 20


def crop_facility_regions(input_folder, output_folder):
    input_path = BASE_DIR / input_folder
    output_path = BASE_DIR / output_folder
    
    # 출력 루트 폴더 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 폴더 내 모든 PNG 이미지 찾기
    image_files = list(input_path.glob("*.png"))
    
    if not image_files:
        print(f"처리할 이미지가 없습니다: {input_path}")
        return
        
    print(f"총 {len(image_files)}장의 이미지에서 시설물 영역 분할 작업을 시작합니다.")
    
    # field_config에 정의된 6개의 시설물 슬롯 (FA-01 ~ FA-06)
    facility_slots = fc.FACILITY_SLOTS # ["FA-01", "FA-02", "FA-03", "FA-04", "FA-05", "FA-06"]
    
    for img_file in image_files:
        print(f"\n[작업 중] {img_file.name}")
        
        # 1. 이미지 로드
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  [!] 이미지 로드 실패: {img_file.name}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # 2. 이미지의 실제 가로 길이(FIELD_WIDTH_CM)를 기준으로 px/cm 스케일 계산
        actual_px_per_cm = img_w / fc.FIELD_WIDTH_CM # 500cm 기준
        
        # 이미지별 하위 폴더 생성 (예: cropped_facilities/augmented_map_0001/)
        image_output_dir = output_path / img_file.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 6개의 시설물 구역 크롭
        for fac_name in facility_slots:
            if fac_name not in fc.SEGMENTS:
                print(f"  [!] {fac_name} 구역 정보가 field_config.py에 없습니다.")
                continue
                
            rect = fc.SEGMENTS[fac_name].copy() # {"x_min", "y_min", "x_max", "y_max"}
            
            # cm 단위를 픽셀 좌표로 환산
            # print(fac_name, rect)
            
            if rect["y_min"] == 0:
                rect["y_max"] += PADDING_LENGTH
            else:
                rect["y_min"] -= PADDING_LENGTH

            x_min_px = int(rect["x_min"] * actual_px_per_cm)
            y_min_px = int(rect["y_min"] * actual_px_per_cm)
            x_max_px = int(rect["x_max"] * actual_px_per_cm)
            y_max_px = int(rect["y_max"] * actual_px_per_cm)
            
            # 이미지 크롭 (OpenCV는 [y, x] 순으로 슬라이싱)
            cropped_fac = img[y_min_px:y_max_px, x_min_px:x_max_px]
            
            # 파일 저장 경로 설정 (예: FA-01.png)
            save_path = image_output_dir / f"{fac_name}.png"
            
            # 무손실 PNG 저장
            cv2.imwrite(str(save_path), cropped_fac, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"  - {fac_name} 저장 완료 -> {save_path.relative_to(BASE_DIR)}")

    print(f"\n모든 작업이 완료되었습니다! 분할된 이미지 확인: {output_path}")

if __name__ == "__main__":
    # 설정: 캘리브레이션된 원본 폴더와 분할 저장할 폴더명
    INPUT_DIR = "reherse_1st_calibrated"       # 캘리브레이션된 원본 맵 폴더
    OUTPUT_DIR = "cropped_facilities"   # 6개씩 쪼개서 저장할 폴더
    
    crop_facility_regions(INPUT_DIR, OUTPUT_DIR)