import os
import gdown

file_id = "1LC2-fgl4uxpjM1aFq9YqGA6QJWj00MNN"
url = f"https://drive.google.com/uc?id={file_id}"

output_path = "checkpoints"  # 저장할 파일명 원하는 걸로
os.makedirs(output_path, exist_ok=True)
gdown.download(url, output_path, quiet=False)