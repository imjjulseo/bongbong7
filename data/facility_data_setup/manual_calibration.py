# aruco가 안되어서 사용한 임시 코드

import cv2
import numpy as np
from pathlib import Path

# 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

# 마우스로 클릭한 4개의 좌표를 저장할 리스트
pts_src = []
scale_factor = 3  # 화면에 띄울 때 축소할 비율 (4K 방지용)

def mouse_click(event, x, y, flags, param):
    global pts_src
    # 마우스 왼쪽 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts_src) < 4:
            # 축소된 화면의 좌표를 원본 해상도 좌표로 복원하여 저장
            real_x, real_y = x * scale_factor, y * scale_factor
            pts_src.append((real_x, real_y))
            print(f"[{len(pts_src)}/4] 좌표 저장됨: ({real_x}, {real_y})")

def manual_calibration(input_image_name, output_image_name):
    global pts_src
    pts_src = []
    input_path = BASE_DIR / input_image_name
    output_path = BASE_DIR / output_image_name

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"에러: 이미지를 찾을 수 없습니다. ({input_path})")
        return

    # 화면에 띄우기 위해 이미지 축소
    h, w = img.shape[:2]
    display_img = cv2.resize(img, (w // scale_factor, h // scale_factor))

    window_name = "Select 4 Corners (TL -> TR -> BR -> BL)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    print("-" * 50)
    print("경기장의 네 모서리를 다음 순서대로 클릭해 줘!")
    print("1. 좌측 상단 (Top-Left)")
    print("2. 우측 상단 (Top-Right)")
    print("3. 우측 하단 (Bottom-Right)")
    print("4. 좌측 하단 (Bottom-Left)")
    print("-" * 50)
    print("※ 4개를 다 찍으면 자동으로 변환 및 저장이 진행돼.")
    print("※ 점을 잘못 찍었다면 'R' 키를 눌러서 리셋할 수 있어.")

    while True:
        temp_img = display_img.copy()
        
        # 클릭한 점들을 화면에 빨간색으로 표시
        for pt in pts_src:
            disp_x, disp_y = pt[0] // scale_factor, pt[1] // scale_factor
            cv2.circle(temp_img, (disp_x, disp_y), 5, (0, 0, 255), -1)

        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF

        # 4개의 점이 모두 찍히면 루프 탈출
        if len(pts_src) == 4:
            cv2.waitKey(500) # 0.5초 대기 후 진행
            break
        
        # 'R' 키를 누르면 점 초기화
        if key == ord('r') or key == ord('R'):
            pts_src = []
            print("좌표가 리셋되었어. 다시 클릭해 줘!")

    cv2.destroyAllWindows()

    if len(pts_src) != 4:
        return

    # -------------------------------------------------------------
    # Perspective Transform (호모그래피) 계산 및 적용
    # -------------------------------------------------------------
    # 대회 규격: 가로 500cm, 세로 400cm
    width_cm = 500
    height_cm = 400
    
    # 1cm당 몇 픽셀로 변환할지 설정 (ex: 10이면 5000x4000 픽셀로 저장됨)
    px_per_cm = 10 
    
    output_width = int(width_cm * px_per_cm)
    output_height = int(height_cm * px_per_cm)

    # 목적지 좌표 (좌상, 우상, 우하, 좌하)
    pts_dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    src_array = np.array(pts_src, dtype=np.float32)

    # 변환 매트릭스 계산
    matrix = cv2.getPerspectiveTransform(src_array, pts_dst)

    print("이미지 워핑(Warping) 진행 중...")
    # 원본 이미지(고화질)에 매트릭스 적용
    warped_img = cv2.warpPerspective(img, matrix, (output_width, output_height))

    # 무손실 PNG로 저장
    cv2.imwrite(str(output_path), warped_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print("\n작업 완료!")
    print(f"저장 경로: {output_path}")
    print(f"결과물 해상도: {output_width} x {output_height} 픽셀")
    print(f"이제 1cm가 정확히 {px_per_cm} 픽셀과 매칭돼!")


def run_calibration_on_all_pngs(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. output 폴더가 없으면 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 모든 하위 폴더에서 .png 파일 찾기
    png_files = list(input_path.rglob("*.png"))
    
    if not png_files:
        print("발견된 png 파일이 없습니다.")
        return

    print(f"총 {len(png_files)}개의 파일을 처리합니다.")

    # 3. 각 파일에 대해 함수 적용 및 저장
    for i, file_path in enumerate(png_files, start=1):
        # 결과 파일 이름을 숫자로 통일 (예: 00001.png)
        new_filename = f"{i:05d}.png"
        target_path = output_path / new_filename
        
        try:
            # 사용자가 정의한 함수 호출
            # input은 원본 경로, output은 저장할 경로
            manual_calibration(file_path, target_path)
            print(f"처리 완료: {file_path.name} -> {new_filename}")
        except Exception as e:
            print(f"처리 실패 ({file_path.name}): {e}")

# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------
if __name__ == "__main__":


    # 불러올 원본 빈 맵 이미지 (마커 인식이 안 된 사진)
    INPUT_FILE = BASE_DIR / "reherse_1st/reherse_1st_uncalibrated"
    
    # 변환되어 저장될 임시 정사영상
    OUTPUT_FILE = BASE_DIR / "reherse_1st/reherse_1st_manual_calibrated"
    
    run_calibration_on_all_pngs(INPUT_FILE, OUTPUT_FILE)