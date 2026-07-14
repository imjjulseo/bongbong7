import cv2
import numpy as np
import random
from pathlib import Path
import sys
import os

# ---------------------------------------------------------
# 1. 경로 및 모듈 설정
# ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.dirname(__file__))

import field_config as fc

BASE_DIR = Path(__file__).resolve().parent

# 객체 외곽선으로부터 구역 경계선까지 띄울 물리적 거리 (5cm)
BOUNDARY_MARGIN_CM = 5.0 

# data.yaml 기준 클래스 맵핑 (nc: 6)[cite: 2]
YOLO_CLASS_MAP = {
    "big": 0,
    "cluster": 1,
    "dumb": 2,
    "medium": 3,
    "missile": 4,
    "small": 5
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

def is_overlapping(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    padding = 10 
    if x1 >= x2 + w2 + padding or x2 >= x1 + w1 + padding:
        return False
    if y1 >= y2 + h2 + padding or y2 >= y1 + h1 + padding:
        return False
    return True

# ---------------------------------------------------------
# 3. 메인 합성 & 라벨링 로직
# ---------------------------------------------------------
def generate_constrained_map(image_output_path, label_output_path, use_lay_variant=False):
    input_bg_path = BASE_DIR / "empty_calibrated.png"
    objects_dir = BASE_DIR / "cropped_objects_final"

    bg_img = cv2.imread(str(input_bg_path))
    if bg_img is None:
        print(f"배경 이미지를 찾을 수 없습니다: {input_bg_path}")
        return

    bg_h, bg_w = bg_img.shape[:2]
    actual_px_per_cm = bg_w / fc.FIELD_WIDTH_CM
    margin_px = int(BOUNDARY_MARGIN_CM * actual_px_per_cm)

    # 공통 배율 도출
    anchor_filename = "big.png"
    anchor_path = objects_dir / anchor_filename
    anchor_img = cv2.imread(str(anchor_path), cv2.IMREAD_UNCHANGED)
    
    if anchor_img is None:
        print(f"기준 파일({anchor_filename})이 없어 배율 계산 실패!")
        return
        
    anchor_cropped = crop_to_visible_alpha(anchor_img)
    anchor_h_px, anchor_w_px = anchor_cropped.shape[:2]
    
    anchor_real_cm = max(fc.CRATER_SIZE_TABLE_MM["big"]["w"], fc.CRATER_SIZE_TABLE_MM["big"]["h"]) / 10.0
    anchor_target_px = anchor_real_cm * actual_px_per_cm
    universal_scale = anchor_target_px / max(anchor_h_px, anchor_w_px)

    crater_files = ["small.png", "medium.png", "big.png"]
    uxo_files = ["cluster.png", "missile.png", "dumb.png"]
    
    if use_lay_variant:
        uxo_files.extend(["cluster_lay.png", "missile_lay.png", "dumb_lay.png"])

    all_zones = list(fc.SEGMENTS.keys())
    rw_tw_zones = [z for z in all_zones if not z.startswith("FA")]
    fa_zones = [z for z in all_zones if z.startswith("FA")]

    crater_target_zones = random.sample(rw_tw_zones, 5)
    uxo_target_zones = random.sample(rw_tw_zones, 6)
    fa_uxo_zones = [z for z in fa_zones if random.random() < 0.5]

    yolo_annotations = []

    for zone_name in all_zones:
        rect_px = get_pixel_zone_rect(zone_name, actual_px_per_cm)
        if not rect_px: continue
        x1, y1, x2, y2 = rect_px
        
        objects_to_place = []
        if zone_name in crater_target_zones:
            objects_to_place.append(random.choice(crater_files))
        if zone_name in uxo_target_zones or zone_name in fa_uxo_zones:
            objects_to_place.append(random.choice(uxo_files))

        if not objects_to_place:
            continue

        placed_boxes = []

        for obj_filename in objects_to_place:
            obj_path = objects_dir / obj_filename
            obj_img = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            if obj_img is None: continue
                
            obj_cropped = crop_to_visible_alpha(obj_img)
            h, w = obj_cropped.shape[:2]
            
            new_w = max(1, int(w * universal_scale))
            new_h = max(1, int(h * universal_scale))
            obj_scaled = cv2.resize(obj_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            angle = random.uniform(0, 360)
            obj_rotated = rotate_image_with_alpha(obj_scaled, angle)
            
            # [핵심 변경] 회전 직후에 생긴 투명 여백을 한 번 더 잘라내어 타이트하게 만듦
            obj_rotated_tight = crop_to_visible_alpha(obj_rotated)
            
            # 타이트해진 최종 이미지의 너비와 높이 사용
            obj_h, obj_w = obj_rotated_tight.shape[:2]
            
            min_x = x1 + margin_px
            min_y = y1 + margin_px
            max_x = x2 - obj_w - margin_px
            max_y = y2 - obj_h - margin_px
            
            if max_x < min_x or max_y < min_y:
                print(f"  [!] {zone_name} 구역은 너무 좁아 {obj_filename} 배치가 불가능합니다.")
                continue
                
            placed = False
            for _ in range(50):
                rand_x = random.randint(min_x, max_x)
                rand_y = random.randint(min_y, max_y)
                current_box = (rand_x, rand_y, obj_w, obj_h)
                
                overlap = any(is_overlapping(current_box, pb) for pb in placed_boxes)
                
                if not overlap:
                    # 타이트하게 잘린 이미지를 배경에 합성
                    bg_img = overlay_transparent(bg_img, obj_rotated_tight, rand_x, rand_y)
                    placed_boxes.append(current_box)
                    placed = True
                    
                    base_name = obj_filename.replace(".png", "").replace("_lay", "")
                    class_id = YOLO_CLASS_MAP.get(base_name, -1)
                    
                    if class_id != -1:
                        # 0~1 사이의 값으로 정규화
                        center_x = (rand_x + obj_w / 2.0) / bg_w
                        center_y = (rand_y + obj_h / 2.0) / bg_h
                        norm_w = obj_w / bg_w
                        norm_h = obj_h / bg_h
                        
                        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                    break
            
            if not placed:
                print(f"  [!] {zone_name} 구역 내에 {obj_filename} 빈자리가 없어 생략되었습니다.")

    cv2.imwrite(str(image_output_path), bg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    with open(label_output_path, "w") as f:
        f.write("\n".join(yolo_annotations))

# ---------------------------------------------------------
if __name__ == "__main__":
    num_images_to_generate = 5
    
    dataset_dir = BASE_DIR / "generated_dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"총 {num_images_to_generate}장의 이미지와 라벨(.txt) 파일 생성을 시작합니다.\n")
    
    for i in range(1, num_images_to_generate + 1):
        file_base = f"augmented_map_{i:04d}"
        
        img_out_path = images_dir / f"{file_base}.png"
        lbl_out_path = labels_dir / f"{file_base}.txt"
        
        print(f"--- [{i}/{num_images_to_generate}] {file_base} 생성 중 ---")
        generate_constrained_map(
            image_output_path=img_out_path, 
            label_output_path=lbl_out_path, 
            use_lay_variant=True
        )
        print("-" * 50)
        
    print(f"\n작업 완료! 모든 데이터는 '{dataset_dir}'에 저장되었습니다.")