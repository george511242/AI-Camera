import glob
import os
import cv2
from pathlib import Path

def save_image_with_meta(save_path: str, file_name: str, frame, meta: str, max_files: int):
    save_image(save_path, file_name + '.jpg', frame)
    save_text_to_file(save_path, file_name + '.meta', meta)
    deleted_file = max_files_cycling(save_path, 'jpg', max_files)
    if deleted_file:
        os.remove(deleted_file.replace('jpg', 'meta'))

def save_image(save_path, file_name, frame):
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(str(Path(save_path).joinpath(file_name)), frame)

def save_text_to_file(save_path, file_name, text):
    with open(f'{save_path}/{file_name}', 'w') as f:
        f.write(text)


def max_files_cycling(folder: str, file_type: str = 'jpg', max_files: int = 5000) -> str:
    # 檢查文件數量，如果超過限制，則刪除最舊的文件
    filtered_files = glob.glob(os.path.join(folder, f"*.{file_type}"))
    if len(filtered_files) > max_files:
        filtered_files.sort(key=os.path.getmtime)  # 按修改時間排序
        os.remove(filtered_files[0])  # 刪除最舊的文件
        return filtered_files[0]
    return None