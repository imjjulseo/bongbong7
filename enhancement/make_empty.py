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

# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------
if __name__ == "__main__":
    # 불러올 원본 빈 맵 이미지 (마커 인식이 안 된 사진)
    INPUT_FILE = "empty.png"
    
    # 변환되어 저장될 임시 정사영상
    OUTPUT_FILE = "empty_topview_manual.png"
    
    manual_calibration(INPUT_FILE, OUTPUT_FILE)



# aruco를 탐지해서 빈 파일 좌표계를 만드는 코드
# aruco가 잘 찍히지 않아 일단은 보류


# import cv2
# from pathlib import Path
# import traceback
# import sys

# current_dir = Path(__file__).resolve().parent
# tools_dir = current_dir.parent / "src"

# # 3. 파이썬이 모듈을 검색하는 경로 리스트(sys.path) 맨 앞에 이 경로를 추가합니다.
# sys.path.insert(0, str(tools_dir))

# # 작성해둔 calibration.py에서 FieldCalibrator 클래스를 불러옵니다.
# # (calibration.py 파일이 같은 폴더에 있다고 가정)
# from calibration import FieldCalibrator

# # 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
# current_file = Path(__file__).resolve()
# BASE_DIR = current_file.parent

# def create_calibrated_empty_map(input_image_name, output_image_name):
#     # 경로 설정
#     input_path = BASE_DIR / input_image_name
#     output_path = BASE_DIR / output_image_name

#     # 1. 빈 맵 이미지 불러오기
#     print(f"이미지 로딩 중: {input_path}")
#     empty_img = cv2.imread(str(input_path))
    
#     if empty_img is None:
#         print(f"에러: 이미지를 찾을 수 없습니다. 경로를 확인하세요: {input_path}")
#         return

#     # 2. Calibrator 객체 생성
#     calibrator = FieldCalibrator()

#     try:
#         # 3. ArUco 마커를 검출하여 호모그래피 매트릭스 계산
#         print("ArUco 마커 탐지 및 캘리브레이션 연산 중...")
#         # refine_subpixel=True 옵션을 통해 서브픽셀 단위로 정밀하게 코너를 찾음
#         calibrator.calibrate_from_image(empty_img, refine_subpixel=True)
#         print("캘리브레이션 완료! 호모그래피 매트릭스 확보 성공.")

#         # 4. 실제 길이를 반영한 Top-view(정사영상)로 워핑(Warping)
#         print("이미지 워핑(Warping) 진행 중...")
#         # px_per_cm을 명시하지 않으면 field_config.py의 WARP_PX_PER_CM을 기본값으로 사용
#         topview_img, px_scale = calibrator.warp_to_topview(empty_img)

#         # 5. 결과물 저장 (화질 보존을 위해 무손실 PNG로 저장)
#         cv2.imwrite(str(output_path), topview_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
#         h, w = topview_img.shape[:2]
#         print(f"\n작업 완료!")
#         print(f"저장 경로: {output_path}")
#         print(f"결과물 해상도: {w} x {h} 픽셀")
#         print(f"스케일 정보: 1cm = {px_scale} 픽셀")
#         print("이제 이 빈 맵의 픽셀 좌표는 실제 경기장의 cm 수치와 완벽히 1:1 대응합니다.")

#     except Exception as e:
#         print("\n[!] 캘리브레이션 중 오류가 발생했습니다.")
#         print(e)
#         traceback.print_exc()

# # ---------------------------------------------------------
# # 실행 영역
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     # 불러올 원본 빈 맵 이미지 파일명
#     INPUT_FILE = "drone_temp.png"
    
#     # 변환되어 저장될 정사영상 이미지 파일명
#     OUTPUT_FILE = "empty_topview.png"
    
#     create_calibrated_empty_map(INPUT_FILE, OUTPUT_FILE)