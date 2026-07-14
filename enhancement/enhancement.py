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
    """회전 후 잘리지 않게 캔버스 크기를 재계산하여 반환"""
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
# 3. 메인 합성 로직
# ---------------------------------------------------------
def generate_constrained_map(output_path, use_lay_variant=False):
    input_bg_path = BASE_DIR / "empty_calibrated.png"
    objects_dir = BASE_DIR / "cropped_objects_final"

    bg_img = cv2.imread(str(input_bg_path))
    if bg_img is None:
        print(f"배경 이미지를 찾을 수 없습니다: {input_bg_path}")
        return

    actual_px_per_cm = bg_img.shape[1] / fc.FIELD_WIDTH_CM
    
    # 5cm에 해당하는 픽셀 수 계산
    margin_px = int(BOUNDARY_MARGIN_CM * actual_px_per_cm)

    # ---------------------------------------------------------
    # 공통 배율(Universal Scale) 도출
    # ---------------------------------------------------------
    anchor_filename = "big.png"
    anchor_path = objects_dir / anchor_filename
    anchor_img = cv2.imread(str(anchor_path), cv2.IMREAD_UNCHANGED)
    
    if anchor_img is None:
        print(f"기준 파일({anchor_filename})이 없어 배율 계산 실패!")
        return
        
    anchor_cropped = crop_to_visible_alpha(anchor_img)
    anchor_h, anchor_w = anchor_cropped.shape[:2]
    
    anchor_real_cm = max(fc.CRATER_SIZE_TABLE_MM["big"]["w"], fc.CRATER_SIZE_TABLE_MM["big"]["h"]) / 10.0
    anchor_target_px = anchor_real_cm * actual_px_per_cm
    
    universal_scale = anchor_target_px / max(anchor_h, anchor_w)

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
            
            # 무작위 회전 적용
            angle = random.uniform(0, 360)
            obj_rotated = rotate_image_with_alpha(obj_scaled, angle)
            
            # 회전 후 넓어진 최종 캔버스의 너비(obj_w)와 높이(obj_h) 확보
            obj_h, obj_w = obj_rotated.shape[:2]
            
            # 객체의 회전된 폭/높이와 5cm 마진을 모두 빼서 배치 가능 영역 설정
            min_x = x1 + margin_px
            min_y = y1 + margin_px
            max_x = x2 - obj_w - margin_px
            max_y = y2 - obj_h - margin_px
            
            if max_x < min_x or max_y < min_y:
                print(f"  [!] {zone_name} 구역은 5cm 여백을 두기엔 너무 좁아 {obj_filename} 배치가 불가능합니다.")
                continue
                
            placed = False
            for _ in range(50):
                rand_x = random.randint(min_x, max_x)
                rand_y = random.randint(min_y, max_y)
                current_box = (rand_x, rand_y, obj_w, obj_h)
                
                overlap = any(is_overlapping(current_box, pb) for pb in placed_boxes)
                
                if not overlap:
                    bg_img = overlay_transparent(bg_img, obj_rotated, rand_x, rand_y)
                    placed_boxes.append(current_box)
                    placed = True
                    break
            
            if not placed:
                print(f"  [!] {zone_name} 구역 내에 {obj_filename}이 들어갈 빈자리가 없어 생략되었습니다.")

    cv2.imwrite(str(output_path), bg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ---------------------------------------------------------
if __name__ == "__main__":
    num_images_to_generate = 5
    output_dir = BASE_DIR / "generated_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"총 {num_images_to_generate}장의 안전거리(5cm) 확보 맵 생성을 시작합니다.\n")
    
    for i in range(1, num_images_to_generate + 1):
        file_name = f"augmented_map_{i:03d}.png"
        out_path = output_dir / file_name
        
        print(f"--- [{i}/{num_images_to_generate}] {file_name} 작업 중 ---")
        generate_constrained_map(output_path=out_path, use_lay_variant=True)
        print("-" * 50)