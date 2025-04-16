import os
# Workaround OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pickle
from insightface import app
from sklearn.metrics.pairwise import cosine_similarity

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\
                   .apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Carga modelo y base de datos
face_app = app.FaceAnalysis(allowed_modules=['detection','recognition'])
face_app.prepare(ctx_id=0, det_size=(640,640))
with open('face_db.pkl','rb') as f:
    face_db = pickle.load(f)

THRESHOLD = 0.4
video_path = r'C:\Users\cesar\Desktop\projects\reconomiento facil\videos\video2.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"No se pudo abrir el vídeo '{video_path}'")
    exit(1)

# Carpeta de detecciones reconocidas (no 'Desconocido')
os.makedirs('detected', exist_ok=True)

# Video de salida completo
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('output_recognition.mp4', fourcc, fps, (w, h))

frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    proc = apply_clahe(frame)
    faces = face_app.get(proc)

    for face_i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)

        emb = face.normed_embedding.reshape(1, -1)
        sims  = [cosine_similarity(emb, db_emb.reshape(1, -1))[0][0]
                 for db_emb in face_db.values()]
        names = list(face_db.keys())
        idx   = int(np.argmax(sims))
        sim   = sims[idx]

        # Solo reconocidos
        if sim > THRESHOLD:
            label = f"{names[idx]} ({sim:.2f})"
            # dibujar y guardar
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            perc = int(sim * 100)
            filename = f"detected/frame{frame_idx:04d}_face{face_i}_{perc}pct.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1

    # Graba el frame completo en el vídeo de salida
    out.write(frame)

print(f"\nProcesados {frame_idx} frames.")
print(f"Guardadas {saved_count} detecciones reconocidas en carpeta 'detected/'.")
print("Vídeo completo anotado: output_recognition.mp4")

cap.release()
out.release()
