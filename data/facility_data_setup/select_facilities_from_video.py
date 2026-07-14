import cv2
import os
import sys
import random
import yaml
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 및 환경 설정
# ---------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = Path(__file__).resolve().parent

# TODO: 작업할 영상 파일 이름 지정
VIDEO_PATH = BASE_DIR / "building_fire.mp4" 
DATASET_DIR = BASE_DIR / "yolo_facility_dataset"

# 프레임 추출 간격 (예: 5프레임마다 1장씩 검사)
FRAME_INTERVAL = 20

# 학습(train)과 검증(val) 데이터 분할 비율
TRAIN_RATIO = 0.8

# 키보드 입력과 클래스 맵핑
CLASS_MAP = {
    ord('1'): "normal",
    ord('2'): "destroy",
    ord('3'): "fire"
}

# ---------------------------------------------------------
# 2. 데이터셋 폴더 구조 및 YAML 생성
# ---------------------------------------------------------
def setup_dataset():
    if DATASET_DIR.exists():
        print(f"[안내] '{DATASET_DIR.name}' 폴더가 이미 존재하여 이어서 저장합니다.")
        
    for split in ["train", "val"]:
        for class_name in CLASS_MAP.values():
            (DATASET_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
            
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_data = {
        "path": str(DATASET_DIR.absolute()),
        "train": "train",
        "val": "val",
        "nc": len(CLASS_MAP),
        "names": list(CLASS_MAP.values())
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

# ---------------------------------------------------------
# 3. 영상 재생 및 라벨링 메인 로직
# ---------------------------------------------------------
def process_video_labeling():
    setup_dataset()
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("\n" + "="*50)
    print(f"🎬 영상 프레임 실시간 추출 & 라벨링 시작")
    print("="*50)
    print("  [1] Normal  (정상 저장)")
    print("  [2] Destroy (손상 저장)")
    print("  [3] Fire    (화재 저장)")
    print("  [Space] 건너뛰기 (버림)")
    print("  [B 또는 Backspace] 뒤로 가기 (실행 취소)")
    print("  [Q 또는 ESC] 저장하고 종료")
    print("="*50 + "\n")

    counts = {"normal": 0, "destroy": 0, "fire": 0, "skipped": 0}
    current_frame_idx = 0
    
    # 실행 취소를 위한 히스토리 스택 (작업 내역 기록)
    history = []
    
    while current_frame_idx < total_frames:
        # 원하는 프레임으로 바로 이동 (건너뛰거나 뒤로가기 할 때 유용)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print("\n영상의 끝에 도달했습니다.")
            break
            
        # 모니터에 맞게 화면 크기 조절
        h, w = frame.shape[:2]
        display_scale = 1080 / h if h > 1080 else 1.0
        display_w, display_h = int(w * display_scale), int(h * display_scale)
        display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        
        # 안내 텍스트 삽입
        total_saved = counts['normal'] + counts['destroy'] + counts['fire']
        info_text = f"Frame: {current_frame_idx}/{total_frames} | Saved: {total_saved}"
        
        cv2.putText(display_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, "1:Normal | 2:Destroy | 3:Fire", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Space:Skip | B:Undo | Q:Quit", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Video Labeler", display_frame)
        
        is_quit = False
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # [1, 2, 3] 분류 저장
            if key in CLASS_MAP:
                class_name = CLASS_MAP[key]
                split = "train" if random.random() < TRAIN_RATIO else "val"
                
                out_filename = f"frame_{current_frame_idx:06d}.png"
                out_path = DATASET_DIR / split / class_name / out_filename
                
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                counts[class_name] += 1
                
                # 실행 취소를 대비해 히스토리에 기록
                history.append({"frame": current_frame_idx, "class": class_name, "path": out_path})
                print(f"  [+] {class_name.upper():<7} 저장 ({split}): {out_filename}")
                
                current_frame_idx += FRAME_INTERVAL
                break
                
            # [Space] 건너뛰기
            elif key == 32: 
                counts["skipped"] += 1
                history.append({"frame": current_frame_idx, "class": "skipped", "path": None})
                print(f"  [-] 버림 (Space): frame_{current_frame_idx:06d}")
                
                current_frame_idx += FRAME_INTERVAL
                break
            
            # [B 또는 Backspace] 뒤로 가기 (Undo)
            elif key in [ord('b'), ord('B'), 8]: 
                if not history:
                    print("  [!] 첫 화면입니다. 더 이상 뒤로 갈 수 없습니다.")
                    # 다시 입력 대기
                else:
                    last_action = history.pop()
                    # 1) 만약 파일이 저장된 동작이었다면 해당 파일을 삭제
                    if last_action["path"] and last_action["path"].exists():
                        last_action["path"].unlink()
                        print(f"  [⟲] 실행 취소: {last_action['path'].name} 삭제됨")
                    else:
                        print(f"  [⟲] 실행 취소: 스킵(버림) 취소됨")
                        
                    # 2) 카운트 원상 복구
                    counts[last_action["class"]] -= 1
                    
                    # 3) 프레임 인덱스를 이전 상태로 되돌림
                    current_frame_idx = last_action["frame"]
                    break # 내부 while 문을 빠져나가서 다시 프레임을 화면에 그림
                
            # [Q 또는 ESC] 종료
            elif key in [ord('q'), ord('Q'), 27]:
                is_quit = True
                break
                
        if is_quit:
            print("\n작업을 중단하고 종료합니다.")
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    total_saved = counts['normal'] + counts['destroy'] + counts['fire']
    print("\n" + "="*50)
    print(f"✅ 라벨링 작업 완료!")
    print(f"  - 총 저장된 사진: {total_saved}장")
    print(f"    (Normal: {counts['normal']}, Destroy: {counts['destroy']}, Fire: {counts['fire']})")
    print(f"  - 버린 사진 수  : {counts['skipped']}장")
    print(f"  - 데이터셋 경로 : {DATASET_DIR.absolute()}")
    print("="*50)

if __name__ == "__main__":
    process_video_labeling()