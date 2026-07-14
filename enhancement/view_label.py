import cv2
import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 설정 및 모듈 임포트
# ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "generated_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = DATASET_DIR / "visualized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# data.yaml 에 정의된 클래스 이름 순서 (nc: 6)[cite: 2]
CLASS_NAMES = ['big', 'cluster', 'dumb', 'medium', 'missile', 'small']

# 클래스별 바운딩 박스 색상 (BGR 포맷)
COLORS = [
    (0, 0, 255),    # 0: big (빨강)
    (0, 255, 0),    # 1: cluster (초록)
    (255, 0, 0),    # 2: dumb (파랑)
    (0, 255, 255),  # 3: medium (노랑)
    (255, 0, 255),  # 4: missile (자주)
    (255, 255, 0)   # 5: small (청록)
]

def visualize_yolo_labels_with_zones():
    image_files = list(IMAGES_DIR.glob("*.png"))
    
    if not image_files:
        print(f"'{IMAGES_DIR}' 폴더에 이미지가 없어!")
        return

    print(f"총 {len(image_files)}장의 이미지에 구역(Zone)과 객체 박스 그리기 시작...")

    for img_path in image_files:
        # 1. 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # 실제 경기장 가로 길이(500cm)[cite: 1]를 이용해 현재 이미지의 스케일(px/cm) 계산
        actual_px_per_cm = img_w / fc.FIELD_WIDTH_CM

        # ---------------------------------------------------------
        # [추가됨] 2. 경기장 구역(Zone) 경계선 그리기
        # ---------------------------------------------------------
        # 객체보다 밑바탕에 깔리도록 먼저 그립니다.
        for zone_name, cm_rect in fc.SEGMENTS.items():
            zx1 = int(cm_rect["x_min"] * actual_px_per_cm)
            zy1 = int(cm_rect["y_min"] * actual_px_per_cm)
            zx2 = int(cm_rect["x_max"] * actual_px_per_cm)
            zy2 = int(cm_rect["y_max"] * actual_px_per_cm)
            
            # 구역 테두리 그리기 (하얀색, 두께 2)
            cv2.rectangle(img, (zx1, zy1), (zx2, zy2), (255, 255, 255), thickness=2)
            
            # 구역 이름(ex: RW-01) 좌상단에 적기
            cv2.putText(img, zone_name, (zx1 + 10, zy1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # ---------------------------------------------------------
        # 3. YOLO 라벨(Bounding Box) 그리기
        # ---------------------------------------------------------
        label_filename = img_path.stem + ".txt"
        label_path = LABELS_DIR / label_filename
        
        if not label_path.exists():
            print(f"  [!] 라벨 누락됨: {label_filename}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # YOLO 정규화 좌표(0~1)를 실제 픽셀 좌표로 역산
            px_center_x = int(center_x * img_w)
            px_center_y = int(center_y * img_h)
            px_w = int(width * img_w)
            px_h = int(height * img_h)
            
            # 박스의 좌상단(x1, y1)과 우하단(x2, y2) 계산
            x1 = int(px_center_x - px_w / 2)
            y1 = int(px_center_y - px_h / 2)
            x2 = int(px_center_x + px_w / 2)
            y2 = int(px_center_y + px_h / 2)
            
            color = COLORS[class_id % len(COLORS)]
            label_text = CLASS_NAMES[class_id]
            
            # 사각형 그리기 (객체 박스는 눈에 띄게 두께 5로 설정)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=5)
            
            # 텍스트 배경 및 글씨 그리기
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, thickness=-1)
            cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

        # 4. 결과 저장
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  [+] 시각화 완료: {img_path.name} (객체 {len(lines)}개)")

    print(f"\n모든 시각화 작업 완료! '{OUTPUT_DIR}' 폴더를 확인해 봐.")

# ---------------------------------------------------------
if __name__ == "__main__":
    visualize_yolo_labels_with_zones()