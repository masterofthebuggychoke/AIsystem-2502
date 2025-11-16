
from typing import Any, List
import numpy as np
import cv2
from insightface.app import FaceAnalysis

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

    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    Full face similarity pipeline:
    1. Decode input bytes.
    2. Detect faces.
    3. Choose largest face.
    4. Extract keypoints and align.
    5. Compute embeddings.
    6. Return cosine similarity.
    """
    def decode_image(image_bytes):
        img_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def select_largest_face(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def estimate_affine_matrix(kps):
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        dst_pts = np.array(kps, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(dst_pts, ref_pts, method=cv2.LMEDS)
        return M

    # Step 1: Decode images
    img_a = decode_image(image_a)
    img_b = decode_image(image_b)

    # Step 2: Detect faces
    faces_a = detect_faces(image_a)
    faces_b = detect_faces(image_b)

    if len(faces_a) == 0 or len(faces_b) == 0:
        raise ValueError("Face not found in one or both images")

    # Step 3: Select largest face
    face_a = select_largest_face(faces_a)
    face_b = select_largest_face(faces_b)

    print("Num faces in A:", len(faces_a))
    print("Num faces in B:", len(faces_b))
    print("Largest face A bbox:", face_a.bbox)
    print("Keypoints A:", face_a.kps)

    # Step 4: Keypoint extraction + alignment
    M_a = estimate_affine_matrix(face_a.kps)
    M_b = estimate_affine_matrix(face_b.kps)

    print("Affine matrix A:", M_a)
    print("Affine matrix B:", M_b)

    aligned_a = warp_face(img_a, M_a)
    aligned_b = warp_face(img_b, M_b)

    print("Aligned face A shape:", aligned_a.shape)
    print("Aligned face B shape:", aligned_b.shape)
    # Step 5: Use precomputed embeddings from detected faces
    emb_a = face_a.embedding
    emb_b = face_b.embedding



    # Step 6: Cosine similarity
    similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    return similarity