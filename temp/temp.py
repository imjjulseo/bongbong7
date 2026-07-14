from pathlib import Path

current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent

def rename_files_sequentially(folder_path, extension=".png"):
    """
    폴더 내의 특정 확장자 파일들을 1, 2, 3... 으로 이름을 바꿉니다.
    """
    folder = Path(folder_path)
    
    # 1. 해당 폴더에서 지정한 확장자를 가진 파일들만 골라내어 정렬 (중요!)
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() == extension.lower()])
    
    if not files:
        print(f"폴더 내에 {extension} 파일이 없습니다.")
        return

    # 2. 순서대로 이름 변경
    for i, file_path in enumerate(files, start=1):
        # 새로운 이름 생성 (예: 1.jpg, 2.jpg...)
        new_name = f"{i}{extension}"
        new_path = folder / new_name
        
        # 파일 이름 변경
        file_path.rename(new_path)
        print(f"{file_path.name} -> {new_name}")

# --- 사용 방법 ---
# 작업할 폴더 경로를 입력하세요
target_folder = BASE_DIR / "." 
rename_files_sequentially(target_folder, extension=".png")