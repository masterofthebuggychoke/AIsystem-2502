
from typing import Any, List
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# 한 사진에 얼굴이 여러개 있으면, 일단 인식이 가장 용이하게끔, 가장 큰 얼굴 하나만 선택하는 방식을 채택하였습니다. 
# --- Global face analysis model ---
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode
    return face_app


def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.
    Expects raw image bytes or a decoded image (NumPy array).
    Returns a list of face objects (bbox, kps, embedding, etc.).
    """
    app = get_face_app()

    # If input is bytes, decode it
    if isinstance(image, bytes):
        img_arr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Convert to RGB if needed (InsightFace expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    return faces


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints from a given face image or bounding box crop.
    Returns 5-point landmark coordinates.
    """
    app = get_face_app()
    faces = app.get(face_image)
    if len(faces) == 0:
        return None
    return faces[0].kps


def warp_face(image: Any, homography_matrix: Any) -> Any:

    return cv2.warpAffine(image, homography_matrix, (112, 112))


def compute_face_embedding(face_image: np.ndarray) -> np.ndarray:

    app = get_face_app()
    faces = app.get(face_image)
    if len(faces) == 0 or not hasattr(faces[0], "embedding"):
        raise ValueError("Failed to compute embedding from face image")
    return faces[0].embedding


def antispoof_check(face_image: Any) -> float:
    """
    Simple heuristic anti-spoofing:
    Return a 'spoof score' where lower means more likely real.
    여기서는 선명도 + 밝기를 기반으로 대충 점수를 만든다.
    """

    # 1. bytes면 디코딩
    if isinstance(face_image, bytes):
        img_arr = np.frombuffer(face_image, np.uint8)
        face_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # 3. 선명도 (variance of Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4. 밝기
    mean_brightness = gray.mean()

    # 5. 선명도 기반 점수 (0~1, 값이 클수록 더 선명 → 진짜일 가능성↑)
    focus_score = np.clip(lap_var / 500.0, 0.0, 1.0)

    # 6. 밝기 기반 점수 (너무 어둡거나 밝으면 점수↓)
    brightness_score = 1.0 - np.clip(abs(mean_brightness - 130) / 130.0, 0.0, 1.0)

    # 7. 0~1 사이의 “실제 같음 점수”
    live_score = 0.7 * focus_score + 0.3 * brightness_score

    # 함수 설명과 맞추려면: lower is more likely to be real
    # → live_score가 클수록 real이니까, 1 - live_score로 변환
    spoof_score = 1.0 - live_score

    return float(spoof_score)


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    Full face similarity pipeline (multi-face version):
    1. Decode input bytes.
    2. Detect all faces in each image.
    3. For each face, run anti-spoofing and get embedding.
    4. Compute cosine similarity for every (A_i, B_j) pair.
    5. Return the maximum similarity among all valid pairs.
    """
    def decode_image(image_bytes):
        img_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 얼굴 bbox 크롭용
    def crop_face(img, face):
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        return img[y1:y2, x1:x2]

    # 1) 이미지 디코딩 (RGB)
    img_a = decode_image(image_a)
    img_b = decode_image(image_b)

    # 2) 얼굴 검출 (각 사진에서 여러 얼굴)
    faces_a = detect_faces(image_a)
    faces_b = detect_faces(image_b)

    print("Num faces in A:", len(faces_a))
    print("Num faces in B:", len(faces_b))

    if len(faces_a) == 0 or len(faces_b) == 0:
        raise ValueError("Face not found in one or both images")

    # 3) 각 얼굴별로 antispoof + embedding 계산
    THRESH = 0.9  # spoof score threshold (높을수록 가짜일 가능성↑)

    valid_a = []  # (index, face_obj, embedding)
    for i, f in enumerate(faces_a):
        face_crop = crop_face(img_a, f)
        spoof_score = antispoof_check(face_crop)
        print(f"[A-{i}] spoof score:", spoof_score)
        if spoof_score > THRESH:
            print(f"[A-{i}] skipped due to high spoof score")
            continue
        emb = f.embedding
        valid_a.append((i, f, emb))

    valid_b = []
    for j, f in enumerate(faces_b):
        face_crop = crop_face(img_b, f)
        spoof_score = antispoof_check(face_crop)
        print(f"[B-{j}] spoof score:", spoof_score)
        if spoof_score > THRESH:
            print(f"[B-{j}] skipped due to high spoof score")
            continue
        emb = f.embedding
        valid_b.append((j, f, emb))

    if len(valid_a) == 0 or len(valid_b) == 0:
        raise ValueError("No valid (non-spoof) faces found in one or both images")

    # 4) 모든 (A_i, B_j) 쌍에 대해 코사인 유사도 계산
    best_sim = -1.0
    best_pair = None

    for i, face_a, emb_a in valid_a:
        for j, face_b, emb_b in valid_b:
            sim = float(
                np.dot(emb_a, emb_b) /
                (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
            )
            print(f"Similarity A-{i} vs B-{j}:", sim)

            if sim > best_sim:
                best_sim = sim
                best_pair = (i, j)

    print("Best pair:", best_pair, "Best similarity:", best_sim)
    return best_sim