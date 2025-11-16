import numpy as np
from util import calculate_face_similarity

# 두 이미지 파일 경로 지정
image_path_a = "people1.jpg"
image_path_b = "people2.jpg"

# 이미지 파일을 바이너리로 읽음
with open(image_path_a, "rb") as f:
    image_bytes_a = f.read()

with open(image_path_b, "rb") as f:
    image_bytes_b = f.read()

# 유사도 계산
try:
    similarity = calculate_face_similarity(image_bytes_a, image_bytes_b)
    print(f"Face similarity score: {similarity:.4f}")
except Exception as e:
    print(f"Error occurred: {e}")
