import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from insightface import app
from sklearn.preprocessing import normalize
import pickle

def apply_clahe(img):
    """Aplica CLAHE al canal de luminancia para mejorar contraste."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def align_face(img, kps):
    """
    Alinea la cara según los ojos (kps[0], kps[1]).
    """
    left_eye, right_eye = kps[0], kps[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calcula centro de los ojos y conviértelo a Python int
    cx, cy = np.mean([left_eye, right_eye], axis=0)
    center = (int(cx), int(cy))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

# 1) Carga modelo
face_app = app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 para CPU

# 2) Carpeta de fotos: cada subcarpeta = una persona
DATA_DIR = r'C:\Users\cesar\Desktop\projects\reconomiento facil\face_photos'


db = {}

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir): continue
    embeddings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # preprocesamiento de iluminación
        img_clahe = apply_clahe(img)

        # detección y extracción
        faces = face_app.get(img_clahe)
        if not faces: continue
        face = max(faces, key=lambda x: x.bbox[2]-x.bbox[0])

        # alineación facial
        aligned = align_face(img_clahe, face.kps)

        # reextraer embedding en rostro alineado
        face_aligned = face_app.get(aligned)[0]
        emb = face_aligned.normed_embedding
        embeddings.append(emb)

    if embeddings:
        db[person] = np.mean(embeddings, axis=0)  # embedding promedio

# 3) Serializar la DB
with open('face_db.pkl', 'wb') as f:
    pickle.dump(db, f)

print(f"Guardada base de datos con {len(db)} identidades.")
