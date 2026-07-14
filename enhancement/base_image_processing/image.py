import cv2
import numpy as np
from pathlib import Path

# 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

def remove_background_and_save(input_dir, output_dir):
    """
    폴더 내의 이미지들의 배경을 지우고 투명한 PNG로 저장하는 함수
    """
    # 결과물을 저장할 폴더 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # input_dir 안의 모든 jpg 파일 검색
    image_paths = list(input_dir.glob("*.png"))
    
    if not image_paths:
        print(f"'{input_dir}' 폴더에 처리할 이미지가 없어!")
        return

    print(f"총 {len(image_paths)}장의 이미지 누끼 따기 시작...")

    for img_path in image_paths:
        # 1. 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 2. 그레이스케일 변환 및 가우시안 블러 (노이즈 제거)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Otsu 이진화 알고리즘 적용
        # 객체(폭파구, 불발탄 등)가 아스팔트 배경보다 어둡다고 가정하여 THRESH_BINARY_INV 사용
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. 모폴로지 연산 (마스크 내부의 구멍을 메우고 자잘한 노이즈 제거)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5. 윤곽선(Contours) 찾기
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"객체를 찾지 못함: {img_path.name}")
            continue

        # 가장 넓이가 큰 윤곽선 하나만 선택 (근접 촬영된 타겟 객체)
        largest_contour = max(contours, key=cv2.contourArea)

        # 6. 빈 마스크 생성 후 가장 큰 윤곽선 내부를 흰색(255)으로 채우기
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 7. 원본 이미지에 알파(투명도) 채널 추가
        # B, G, R 채널을 분리한 뒤, mask를 알파 채널로 사용하여 병합
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask])

        # 8. PNG로 저장 (투명도를 유지하려면 확장자가 반드시 .png 여야 함)
        # 파일명은 원본과 동일하게 하되 확장자만 변경
        output_filename = img_path.stem + "_cutout.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), rgba)

    print(f"작업 완료! 누끼 딴 이미지들이 '{output_dir}'에 저장되었어.")


def advanced_remove_background(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob("*.png"))
    
    if not image_paths:
        print(f"'{input_dir}' 폴더에 처리할 이미지가 없어!")
        return

    print(f"총 {len(image_paths)}장의 이미지 정밀 누끼 따기 시작 (GrabCut 적용)...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        
        # 1. GrabCut을 위한 초기 마스크 및 모델 배열 생성
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # 2. 관심 영역(Rect) 설정
        # 우리가 크롭한 이미지의 테두리 3픽셀 정도는 무조건 '배경(아스팔트)'이라고 알고리즘에 알려줌
        margin = 5
        # 예외 처리: 크롭된 이미지가 너무 작으면 margin을 1로 축소
        if w <= margin*2 or h <= margin*2:
            margin = 1
            
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        try:
            # 3. GrabCut 실행 (5번 반복하며 정밀도 향상)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            print(f"[{img_path.name}] GrabCut 처리 중 오류 발생: {e}")
            continue

        # GrabCut 결과 마스크 변환 (0, 2는 배경 / 1, 3은 전경)
        # 배경은 0, 객체는 1로 만드는 작업
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 4. 침식(Erosion) 연산으로 찌꺼기 테두리 깎아내기
        # 너무 많이 깎이면 iterations를 0으로 줄이거나, 덜 깎이면 2로 늘려봐
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1) 

        # 5. 원본 이미지에 알파(투명도) 채널 추가
        # 마스크 값이 1인 곳은 255(완전 불투명), 0인 곳은 0(완전 투명)으로 변환
        b, g, r = cv2.split(img)
        alpha = mask2 * 255
        rgba = cv2.merge([b, g, r, alpha])

        # 6. PNG로 저장
        output_filename = img_path.stem + "_cutout.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"작업 완료! 정밀하게 누끼 딴 이미지들이 '{output_dir}'에 저장되었어.")


def refine_manual_cutout(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob("*.png"))
    
    if not image_paths:
        print(f"'{input_dir}' 폴더에 처리할 이미지가 없어!")
        return

    print(f"총 {len(image_paths)}장의 수동 누끼 이미지 정밀 다듬기 시작...")

    for img_path in image_paths:
        # [핵심] cv2.IMREAD_UNCHANGED 옵션을 줘야 투명도(Alpha) 채널 4개가 모두 유지됨
        img_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img_bgra is None or img_bgra.shape[2] != 4:
            print(f"  [!] '{img_path.name}' 파일에 투명도 채널이 없어 스킵합니다.")
            continue

        # BGR 색상 채널과 Alpha 투명도 채널 분리
        img_bgr = img_bgra[:, :, :3]
        alpha = img_bgra[:, :, 3]

        h, w = img_bgr.shape[:2]
        
        # 1. GrabCut을 위한 초기 마스크 및 모델 생성
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # 2. 마스크 초기화 (알고리즘에 힌트 주기)
        # 수동으로 누끼를 따서 투명해진 부분(alpha == 0)은 '확실한 배경(GC_BGD)'으로 지정
        mask[alpha == 0] = cv2.GC_BGD
        # 수동으로 따서 남아있는 부분(alpha > 0)은 '전경일 가능성 높음(GC_PR_FGD)'으로 지정
        mask[alpha > 0] = cv2.GC_PR_FGD

        try:
            # 3. 박스(Rect) 대신 마스크(Mask) 기반으로 GrabCut 실행
            cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            print(f"[{img_path.name}] GrabCut 처리 중 오류 발생: {e}")
            continue

        # 4. GrabCut 결과 변환 (확실한 배경 0, 전경 1)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 5. 침식(Erosion) 연산으로 남은 찌꺼기 미세하게 깎아내기
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1) 

        # 6. 최종 투명도 병합 후 저장
        b, g, r = cv2.split(img_bgr)
        final_alpha = mask2 * 255
        rgba = cv2.merge([b, g, r, final_alpha])

        output_filename = img_path.stem + "_refined.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"  [+] 정밀 다듬기 완료: {output_filename}")

    print("작업 완료! 정밀하게 다듬어진 이미지들이 저장되었어.")



# # ---------------------------------------------------------
# # 실행 예시
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     # 이전 단계에서 프레임을 저장했던 폴더
#     INPUT_FOLDER = BASE_DIR / "cropped_objects" 
    
#     # 누끼 딴 결과물(PNG)을 저장할 새로운 폴더
#     OUTPUT_FOLDER = BASE_DIR / "cutout_objects" 
    
#     remove_background_and_save(INPUT_FOLDER, OUTPUT_FOLDER)

# 누끼따기 - advanced 버전

# ---------------------------------------------------------
# if __name__ == "__main__":
#     INPUT_FOLDER = BASE_DIR / "cutout_objects_manual" 
#     OUTPUT_FOLDER = BASE_DIR / "cutout_objects_manual_adv" 
    
#     advanced_remove_background(INPUT_FOLDER, OUTPUT_FOLDER)


# ---------------------------------------------------------
if __name__ == "__main__":
    # 수동 누끼 딴 이미지들이 있는 폴더
    INPUT_FOLDER = BASE_DIR / "cutout_objects_manual" 
    
    # GrabCut으로 테두리를 쫙 다듬은 최종 결과를 저장할 폴더
    OUTPUT_FOLDER = BASE_DIR / "cutout_objects_final" 
    
    refine_manual_cutout(INPUT_FOLDER, OUTPUT_FOLDER)