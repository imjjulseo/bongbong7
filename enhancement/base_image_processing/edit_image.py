import cv2
from pathlib import Path
import numpy as np

# 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

def manual_crop_objects_png_resized(input_dir, output_dir, scale_factor=3):
    """
    화면 크기에 맞춰 이미지를 축소하여 띄우고, 
    마우스로 ROI 지정 후 원본 해상도에서 무손실 크롭하여 저장하는 함수
    
    :param scale_factor: 화면에 띄울 때 축소할 비율 (기본값 3: 1/3 사이즈로 띄움)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.png"))
    if not image_paths:
        print(f"'{input_dir}' 폴더에 처리할 PNG 이미지가 없어!")
        return

    print(f"총 {len(image_paths)}장의 PNG 이미지 크롭 작업을 시작합니다. (화면 크기 1/{scale_factor} 축소 모드)")
    print("-" * 50)
    print("[조작 방법]")
    print("1. 마우스 좌클릭 & 드래그 : 축소된 화면에서 객체 박스 그리기")
    print("2. Enter 또는 Space : 그린 박스 확정 (원본 크기로 자동 계산되어 저장됨)")
    print("3. C 키 (또는 박스 없이 Enter) : 현재 이미지 크롭 종료 -> 다음 이미지로 넘어감")
    print("-" * 50)

    for img_path in image_paths:
        # 원본 이미지 읽기 (예: 4K 해상도)
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            continue

        # 화면에 띄우기 위해 1/3 크기로 리사이즈된 이미지 생성
        height, width = original_img.shape[:2]
        display_img = cv2.resize(original_img, (width // scale_factor, height // scale_factor))

        crop_count = 0
        window_name = f"Cropping (1/{scale_factor} view): {img_path.name} (Press 'C' to skip)"

        while True:
            # 리사이즈된 이미지에서 ROI 선택
            bbox = cv2.selectROI(window_name, display_img, fromCenter=False, showCrosshair=True)
            
            x, y, w, h = bbox

            # w와 h가 0보다 크면(정상적으로 박스를 그렸다면)
            if w > 0 and h > 0:
                # 축소된 화면에서 얻은 좌표를 다시 원본 비율로 복원
                orig_x = x * scale_factor
                orig_y = y * scale_factor
                orig_w = w * scale_factor
                orig_h = h * scale_factor

                # 원본 고화질 이미지에서 크롭
                cropped_img = original_img[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                
                output_filename = f"{img_path.stem}_crop_{crop_count:03d}.png"
                output_path = output_dir / output_filename
                
                # 원본 화질 그대로 무손실 저장
                cv2.imwrite(str(output_path), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"  [+] 저장 완료: {output_filename} (실제 크롭 크기: {orig_w}x{orig_h})")
                crop_count += 1
            else:
                print(f"  [-] '{img_path.name}' 작업 종료. 다음 이미지로 넘어갑니다.")
                break
        
        cv2.destroyAllWindows()

    print("모든 PNG 크롭 작업이 완료되었습니다!")



# 마우스 클릭 좌표를 저장할 전역 리스트
points = []

def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리 함수"""
    global points
    # 왼쪽 버튼을 클릭할 때마다 좌표 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def manual_polygon_cutout(input_dir, output_dir):
    global points
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob("*.png"))
    
    if not image_paths:
        print(f"'{input_dir}' 폴더에 처리할 이미지가 없어!")
        return

    print(f"총 {len(image_paths)}장의 이미지 수동 누끼 따기 시작...")
    print("-" * 50)
    print("[조작 방법]")
    print("1. 마우스 좌클릭 : 객체 외곽선을 따라 점 찍기")
    print("2. Z 키 : 방금 찍은 점 취소 (Undo)")
    print("3. Enter 또는 Space : 다각형 완성 및 누끼 저장")
    print("4. C 키 : 현재 이미지 작업 취소 후 다음 이미지로 넘어가기")
    print("-" * 50)

    window_name = "Manual Polygon Cutout"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        points = []  # 이미지마다 점 리스트 초기화

        while True:
            display_img = img.copy()
            
            # 찍힌 점들과 선을 화면에 표시
            if len(points) > 0:
                # 점들을 이어주는 초록색 선 그리기
                cv2.polylines(display_img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
                # 각 점의 위치에 빨간색 동그라미 표시
                for p in points:
                    cv2.circle(display_img, p, radius=3, color=(0, 0, 255), thickness=-1)

            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Enter(13) 또는 Space(32) 누르면 작업 완료
            if key == 13 or key == 32:
                if len(points) >= 3:  # 다각형을 만들려면 최소 3개의 점이 필요함
                    break
                else:
                    print("  [!] 최소 3개의 점을 찍어야 해!")
            
            # 'z' 키를 누르면 마지막으로 찍은 점 실행 취소
            elif key == ord('z'):
                if len(points) > 0:
                    points.pop()
            
            # 'c' 키를 누르면 이 이미지는 건너뜀
            elif key == ord('c'):
                points = []
                break

        # 정상적으로 점을 다 찍고 Enter를 눌렀을 경우
        if len(points) >= 3:
            # 1. 원본 크기의 빈 검은색 마스크 생성
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # 2. 찍은 점들을 연결한 다각형 내부를 흰색(255)으로 채우기
            cv2.fillPoly(mask, [np.array(points)], 255)
            
            # 3. 마스크를 이용해 알파(투명도) 채널 병합
            b, g, r = cv2.split(img)
            rgba = cv2.merge([b, g, r, mask])
            
            # 4. 파일 저장
            output_filename = img_path.stem + "_manual.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"  [+] 저장 완료: {output_filename}")
        else:
            print(f"  [-] '{img_path.name}' 스킵됨.")

    cv2.destroyAllWindows()
    print("모든 수동 누끼 작업이 완료되었습니다!")

# 이미지 crop 코드
# # ---------------------------------------------------------
# # 실행 예시
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     INPUT_FOLDER = BASE_DIR / "extracted_dataset" 
#     OUTPUT_FOLDER = BASE_DIR / "cropped_objects" 
    
#     # scale_factor=3 으로 설정하여 1/3 사이즈로 띄움
#     manual_crop_objects_png_resized(INPUT_FOLDER, OUTPUT_FOLDER, scale_factor=3)

# ---------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_FOLDER = BASE_DIR / "cutout_objects_advanced" 
    OUTPUT_FOLDER = BASE_DIR / "cutout_objects_manual" 
    
    manual_polygon_cutout(INPUT_FOLDER, OUTPUT_FOLDER)

