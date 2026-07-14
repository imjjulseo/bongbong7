import cv2
from pathlib import Path

# 현재 파이썬 파일(.py)의 절대 경로 및 프로젝트 루트 폴더 설정
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

def extract_frames_png(video_path, output_dir, frame_interval=5):
    """
    비디오에서 특정 간격으로 프레임을 추출하여 무손실 PNG 이미지로 저장하는 함수
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"에러: 비디오 파일을 열 수 없어. 경로를 확인해 줘. ({video_path})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0

    print(f"PNG 프레임 추출 시작... (저장 간격: {frame_interval} 프레임)")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # 확장자를 .png로 변경
            output_path = output_dir / f"frame_{frame_count:05d}.png"
            
            # cv2.IMWRITE_PNG_COMPRESSION: 0(압축 안함, 속도 가장 빠름, 용량 큼) ~ 9(최대 압축)
            # 최고 화질과 무손실 처리를 위해 0으로 설정
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"작업 완료! 총 {saved_count}장의 최고 화질 PNG 이미지가 '{output_dir}' 폴더에 저장되었어.")

# ---------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------
if __name__ == "__main__":
    VIDEO_FILE = BASE_DIR / "drone_empty.mp4" 
    OUTPUT_FOLDER = BASE_DIR / "empty_dataset_png" 
    
    extract_frames_png(VIDEO_FILE, OUTPUT_FOLDER, frame_interval=5)