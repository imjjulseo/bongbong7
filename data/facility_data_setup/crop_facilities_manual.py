import cv2
import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "yolo_facility_dataset_2/val/normal"       # 원본 사진들이 있는 폴더 (수정해서 사용해!)
OUTPUT_DIR = BASE_DIR / "yolo_facility_manual_dataset_2/train/normal"  # 잘라낸 사진이 저장될 폴더

# 전역 변수 (마우스 콜백에서 사용)
drawing = False
ix, iy = -1, -1
bbox = None
img_copy = None
display_img = None

def draw_roi(event, x, y, flags, param):
    global drawing, ix, iy, bbox, img_copy, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = display_img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_copy = display_img.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # 좌상단 좌표와 너비, 높이 계산 (역방향으로 드래그해도 정상 작동하도록 방어)
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

def process_manual_crop():
    global img_copy, display_img, bbox
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # png, jpg 파일 모두 읽기
    all_images = list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg"))
    
    if not all_images:
        print(f"[!] '{INPUT_DIR}' 폴더에 처리할 이미지가 없어. 경로를 다시 확인해 줘!")
        return

    cv2.namedWindow("Manual Cropper")
    cv2.setMouseCallback("Manual Cropper", draw_roi)
    
    print("\n" + "="*50)
    print("✂️ 마우스 드래그 수동 크롭 툴 시작")
    print("="*50)
    print("  [마우스 드래그] 자를 영역 선택 (초록색 박스)")
    print("  [S] 선택 영역 크롭 및 저장 (Save)")
    print("  [Space 또는 N] 건너뛰기 (Skip)")
    print("  [R] 박스 다시 그리기 (Reset)")
    print("  [B 또는 Backspace] 뒤로 가기 (Undo)")
    print("  [Q 또는 ESC] 종료 (Quit)")
    print("="*50 + "\n")

    history = []
    idx = 0
    total = len(all_images)
    
    # 화면 축소 비율 (정확히 절반)
    SCALE_FACTOR = 0.2
    
    while idx < total:
        img_path = all_images[idx]
        original_img = cv2.imread(str(img_path))
        
        if original_img is None:
            idx += 1
            continue
            
        # 1. 이미지를 정확히 50% 크기로 리사이즈
        h, w = original_img.shape[:2]
        display_w, display_h = int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)
        display_img = cv2.resize(original_img, (display_w, display_h), interpolation=cv2.INTER_AREA)
        
        img_copy = display_img.copy()
        bbox = None
        
        while True:
            # 상태 표시 텍스트 작성
            temp_view = img_copy.copy()
            cv2.putText(temp_view, f"[{idx+1}/{total}] {img_path.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(temp_view, "Drag -> 'S':Save | 'N':Skip | 'B':Undo", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Manual Cropper", temp_view)
            key = cv2.waitKey(20) & 0xFF
            
            # [S] 저장
            if key in [ord('s'), ord('S')]:
                if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
                    bx, by, bw, bh = bbox
                    
                    # 50%로 줄인 화면에서 딴 박스 좌표를 원본 크기(x2)로 되돌림
                    real_x = int(bx / SCALE_FACTOR)
                    real_y = int(by / SCALE_FACTOR)
                    real_w = int(bw / SCALE_FACTOR)
                    real_h = int(bh / SCALE_FACTOR)
                    
                    # 원본 이미지에서 크롭
                    cropped = original_img[real_y:real_y+real_h, real_x:real_x+real_w]
                    
                    out_path = OUTPUT_DIR / f"cropped_{img_path.name}"
                    cv2.imwrite(str(out_path), cropped)
                    
                    # 히스토리에 저장 내역 기록
                    history.append({"idx": idx, "path": out_path})
                    print(f"  [+] 저장 완료: {out_path.name}")
                    idx += 1
                    break
                else:
                    print("  [!] 먼저 마우스로 자를 영역을 네모나게 드래그해 줘!")
            
            # [Space 또는 N] 건너뛰기 (스킵)
            elif key in [32, ord('n'), ord('N')]: 
                history.append({"idx": idx, "path": None}) # 저장한 파일 없이 스킵했다는 기록
                print(f"  [-] 건너뜀 (Skip): {img_path.name}")
                idx += 1
                break
                
            # [R] 다시 그리기
            elif key in [ord('r'), ord('R')]: 
                bbox = None
                img_copy = display_img.copy()
                
            # [B 또는 Backspace] 뒤로 가기 (실행 취소)
            elif key in [ord('b'), ord('B'), 8]: 
                if not history:
                    print("  [!] 첫 사진이야. 더 뒤로 갈 수 없어.")
                else:
                    last_action = history.pop()
                    
                    # 직전에 파일을 저장했었다면 지워버림
                    if last_action["path"] and last_action["path"].exists():
                        last_action["path"].unlink()
                        print(f"  [⟲] 실행 취소: {last_action['path'].name} 삭제됨")
                    else:
                        print(f"  [⟲] 실행 취소: 스킵 취소됨")
                        
                    idx = last_action["idx"]
                    break
                    
            # [Q 또는 ESC] 종료
            elif key in [ord('q'), ord('Q'), 27]: 
                print("\n작업을 중단할게. 수고했어!")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n✅ 모든 사진 작업 완료! 고생했어.")

if __name__ == "__main__":
    process_manual_crop()