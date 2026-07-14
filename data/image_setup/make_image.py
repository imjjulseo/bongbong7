# aruco를 탐지해서 빈 파일 좌표계를 만드는 코드
import cv2
from pathlib import Path
import traceback
import sys

current_dir = Path(__file__).resolve().parent

BASE_DIR = current_dir

tools_dir = current_dir.parent.parent / "src"

# 3. 파이썬이 모듈을 검색하는 경로 리스트(sys.path) 맨 앞에 이 경로를 추가합니다.
sys.path.insert(0, str(tools_dir))

# 작성해둔 calibration.py에서 FieldCalibrator 클래스를 불러옵니다.
# (calibration.py 파일이 같은 폴더에 있다고 가정)
from calibration import FieldCalibrator

# 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

def create_calibrated_map(input_image_name, output_image_name):
    # 경로 설정
    input_path = BASE_DIR / input_image_name
    output_path = BASE_DIR / output_image_name

    # 1. 빈 맵 이미지 불러오기
    print(f"이미지 로딩 중: {input_path}")
    empty_img = cv2.imread(str(input_path))
    
    if empty_img is None:
        print(f"에러: 이미지를 찾을 수 없습니다. 경로를 확인하세요: {input_path}")
        return

    # 2. Calibrator 객체 생성
    calibrator = FieldCalibrator()

    try:
        # 3. ArUco 마커를 검출하여 호모그래피 매트릭스 계산
        print("ArUco 마커 탐지 및 캘리브레이션 연산 중...")
        # refine_subpixel=True 옵션을 통해 서브픽셀 단위로 정밀하게 코너를 찾음
        calibrator.calibrate_from_image(empty_img, refine_subpixel=True)
        print("캘리브레이션 완료! 호모그래피 매트릭스 확보 성공.")

        # 4. 실제 길이를 반영한 Top-view(정사영상)로 워핑(Warping)
        print("이미지 워핑(Warping) 진행 중...")
        # px_per_cm을 명시하지 않으면 field_config.py의 WARP_PX_PER_CM을 기본값으로 사용
        topview_img, px_scale = calibrator.warp_to_topview(empty_img)

        # 5. 결과물 저장 (화질 보존을 위해 무손실 PNG로 저장)
        cv2.imwrite(str(output_path), topview_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        h, w = topview_img.shape[:2]
        print(f"\n작업 완료!")
        print(f"저장 경로: {output_path}")
        print(f"결과물 해상도: {w} x {h} 픽셀")
        print(f"스케일 정보: 1cm = {px_scale} 픽셀")
        print("이제 이 빈 맵의 픽셀 좌표는 실제 경기장의 cm 수치와 완벽히 1:1 대응합니다.")

    except Exception as e:
        print("\n[!] 캘리브레이션 중 오류가 발생했습니다.")
        print(e)
        traceback.print_exc()

# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_DIR = BASE_DIR / "2nd trial"      # 이미지가 들어있는 폴더
    OUTPUT_DIR = BASE_DIR / "2nd trial_calibrated" # 결과물을 저장할 폴더

    # 결과 저장 폴더가 없으면 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 폴더 내 모든 png 파일 탐색 및 루프 실행
    for img_path in INPUT_DIR.glob("*.png"):
        # 입력 파일명과 출력 파일명 지정
        input_filename = img_path.name
        output_filename = f"{img_path.name.removesuffix(".png")}_calibrated.png"

        print(output_filename)
        
        print(f"처리 중: {input_filename} -> {output_filename}")
        
        #에서 정의한 함수를 루프 외부에서 호출
        try:
            create_calibrated_map(img_path, OUTPUT_DIR / output_filename)
        except Exception as e:
            print(f"오류 발생 ({input_filename}): {e}")

    print("모든 이미지 처리가 완료되었습니다.")