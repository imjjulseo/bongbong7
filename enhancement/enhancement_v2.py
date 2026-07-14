import cv2
import numpy as np
import random
from pathlib import Path
import sys
import os

# ---------------------------------------------------------
# 1. 경로 및 설정
# ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc

BASE_DIR = Path(__file__).resolve().parent

# 물리적 거리(cm) 설정: 픽셀이 아닌 실제 거리로 패딩 보장
BOUNDARY_MARGIN_CM = 5.0   # 구역 경계선과 객체 테두리 사이의 최소 여백
OBJECT_PADDING_CM = 2.0    # 객체와 객체 사이의 최소 간격

YOLO_CLASS_MAP = {
    "big": 0, "cluster": 1, "dumb": 2, 
    "medium": 3, "missile": 4, "small": 5
}

# ---------------------------------------------------------
# 2. 이미지 처리 헬퍼 함수
# ---------------------------------------------------------
def crop_to_visible_alpha(image):
    if image.shape[2] != 4:
        return image
    alpha_channel = image[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    return image

def rotate_image_with_alpha(image, angle):
    """투명도를 유지하며 캔버스를 확장해 객체를 회전시킴"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0, 0, 0, 0))
    return rotated

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    bg_x1, bg_x2 = max(0, x), min(bg_w, x + w)
    bg_y1, bg_y2 = max(0, y), min(bg_h, y + h)
    ol_x1, ol_x2 = max(0, -x), min(w, bg_w - x)
    ol_y1, ol_y2 = max(0, -y), min(h, bg_h - y)

    alpha_s = overlay[ol_y1:ol_y2, ol_x1:ol_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[bg_y1:bg_y2, bg_x1:bg_x2, c] = (
            alpha_s * overlay[ol_y1:ol_y2, ol_x1:ol_x2, c] +
            alpha_l * background[bg_y1:bg_y2, bg_x1:bg_x2, c]
        )
    return background

def get_pixel_zone_rect(zone_name, px_per_cm):
    if zone_name not in fc.SEGMENTS:
        return None
    cm_rect = fc.SEGMENTS[zone_name]
    return (
        int(cm_rect["x_min"] * px_per_cm),
        int(cm_rect["y_min"] * px_per_cm),
        int(cm_rect["x_max"] * px_per_cm),
        int(cm_rect["y_max"] * px_per_cm)
    )

def is_overlapping(box1, box2, padding_px):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # 패딩을 더해서 객체 간 여유 공간을 계산
    if x1 >= x2 + w2 + padding_px or x2 >= x1 + w1 + padding_px:
        return False
    if y1 >= y2 + h2 + padding_px or y2 >= y1 + h1 + padding_px:
        return False
    return True

# ---------------------------------------------------------
# 3. 메인 합성 & 라벨링 로직
# ---------------------------------------------------------
def generate_constrained_map(image_output_path, label_output_path, object_pool):
    input_bg_path = BASE_DIR / "empty_calibrated.png"
    bg_img = cv2.imread(str(input_bg_path))
    
    if bg_img is None:
        print(f"배경 이미지를 찾을 수 없습니다: {input_bg_path}")
        return

    bg_h, bg_w = bg_img.shape[:2]
    actual_px_per_cm = bg_w / fc.FIELD_WIDTH_CM
    
    # cm 단위를 실제 픽셀 단위로 변환
    margin_px = int(BOUNDARY_MARGIN_CM * actual_px_per_cm)
    obj_padding_px = int(OBJECT_PADDING_CM * actual_px_per_cm)

    # 구역 분류
    all_zones = list(fc.SEGMENTS.keys())
    rw_tw_zones = [z for z in all_zones if not z.startswith("FA")]
    fa_zones = [z for z in all_zones if z.startswith("FA")]

    crater_classes = ["big", "medium", "small"]
    uxo_classes = ["cluster", "dumb", "missile"]

    # ---------------------------------------------------------
    # [조건 할당] 객체를 어느 구역에 넣을지 미리 분배
    # ---------------------------------------------------------
    zone_assignments = {z: [] for z in all_zones}

    # 1. 활주로/유도로(RW/TW) 구역에 폭파구 총 5개 무작위 분배
    for z in random.choices(rw_tw_zones, k=5):
        zone_assignments[z].append(random.choice(crater_classes))

    # 2. 활주로/유도로(RW/TW) 구역에 불발탄 총 6개 무작위 분배
    for z in random.choices(rw_tw_zones, k=6):
        zone_assignments[z].append(random.choice(uxo_classes))

    # 3. 시설물(FA) 구역에는 불발탄을 자유롭게 배치 (구역당 0~2개 무작위)
    for z in fa_zones:
        num_uxos = random.randint(0, 0)
        for _ in range(num_uxos):
            zone_assignments[z].append(random.choice(uxo_classes))

    yolo_annotations = []

    for zone_name, assigned_classes in zone_assignments.items():
        if not assigned_classes:
            continue
            
        rect_px = get_pixel_zone_rect(zone_name, actual_px_per_cm)
        if not rect_px: continue
        x1, y1, x2, y2 = rect_px
        
        placed_boxes = []

        for cls_name in assigned_classes:
            if not object_pool.get(cls_name):
                continue
            # 해당 클래스의 누끼 이미지 중 하나를 랜덤 선택
            obj_path = random.choice(object_pool[cls_name])
            obj_img = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            if obj_img is None: continue
                
            # Roboflow 추출 이미지는 이미 스케일이 맞으므로 리사이즈 생략!
            obj_cropped = crop_to_visible_alpha(obj_img)
            
            # 무작위 회전 적용
            angle = random.uniform(0, 360)
            obj_rotated = rotate_image_with_alpha(obj_cropped, angle)
            
            # 회전 후 넓어진 투명 여백을 다시 타이트하게 잘라냄 (핵심!)
            obj_rotated_tight = crop_to_visible_alpha(obj_rotated)
            obj_h, obj_w = obj_rotated_tight.shape[:2]
            
            # 타이트한 객체 테두리 기준으로 margin_px(5cm) 적용
            min_x = x1 + margin_px
            min_y = y1 + margin_px
            max_x = x2 - obj_w - margin_px
            max_y = y2 - obj_h - margin_px
            
            if max_x < min_x or max_y < min_y:
                print(f"  [!] {zone_name} 구역은 너무 좁아 {cls_name} 배치가 생략됩니다.")
                continue
                
            placed = False
            for _ in range(50):
                rand_x = random.randint(min_x, max_x)
                rand_y = random.randint(min_y, max_y)
                current_box = (rand_x, rand_y, obj_w, obj_h)
                
                # cm 기반으로 계산된 obj_padding_px(2cm)를 사용해 겹침 확인
                overlap = any(is_overlapping(current_box, pb, obj_padding_px) for pb in placed_boxes)
                
                if not overlap:
                    bg_img = overlay_transparent(bg_img, obj_rotated_tight, rand_x, rand_y)
                    placed_boxes.append(current_box)
                    placed = True
                    
                    class_id = YOLO_CLASS_MAP.get(cls_name, -1)
                    if class_id != -1:
                        # 중심 좌표 및 너비/높이 정규화(0~1)
                        center_x = (rand_x + obj_w / 2.0) / bg_w
                        center_y = (rand_y + obj_h / 2.0) / bg_h
                        norm_w = obj_w / bg_w
                        norm_h = obj_h / bg_h
                        
                        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                    break
            
            if not placed:
                print(f"  [!] {zone_name} 내에 {cls_name}을 배치할 빈 공간이 부족합니다.")

    cv2.imwrite(str(image_output_path), bg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    with open(label_output_path, "w") as f:
        f.write("\n".join(yolo_annotations))

# ---------------------------------------------------------
if __name__ == "__main__":
    num_images_to_generate = 10
    
    dataset_dir = BASE_DIR / "generated_dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 추출된 객체 이미지(png)들을 클래스별로 미리 로드
    objects_dir = BASE_DIR / "generate_object_image/extracted_objects_advanced"
    object_pool = {}
    for cls_name in YOLO_CLASS_MAP.keys():
        cls_folder = objects_dir / cls_name
        if cls_folder.exists():
            object_pool[cls_name] = list(cls_folder.glob("*.png"))
    
    print(f"총 {num_images_to_generate}장의 조건부 증강 맵 생성을 시작합니다.\n")
    
    for i in range(1, num_images_to_generate + 1):
        file_base = f"augmented_map_{i:04d}"
        img_out_path = images_dir / f"{file_base}.png"
        lbl_out_path = labels_dir / f"{file_base}.txt"
        
        print(f"--- [{i}/{num_images_to_generate}] {file_base} 합성 중 ---")
        generate_constrained_map(img_out_path, lbl_out_path, object_pool)
        
    print(f"\n작업 완료! 모든 데이터는 '{dataset_dir}'에 저장되었습니다.")