"""
AU Detector — Action Units (FACS) en tiempo real
YOLOv8 + OpenCV

NOTA: Este script es un MOCKUP / prototipo demostrativo.
Las Action Units son aproximaciones heurísticas basadas en
Haar Cascades de OpenCV, NO valores reales de un modelo FACS
entrenado (como OpenFace o py-feat). Sirve como prueba de
concepto visual para validar el pipeline antes de integrar
un modelo dedicado.

Detección de caras: YOLOv8 (clase 0 = persona, bbox facial)
combinado con Haar Cascade para región ocular dentro del ROI.

Controles:
    Q  o clic en [SALIR] — cierra la ventana de cámara

Uso:
    python au_detector.py           # webcam
    python au_detector.py video.mp4
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO


MODEL = "yolov8n.pt"

# Descripción de cada AU mostrada en pantalla
AU_INFO = {
    "AU4":  ("Brow Lowerer",   "ceno fruncido / tension"),
    "AU7":  ("Lid Tightener",  "tension en parpado superior"),
    "AU12": ("Lip Corner",     "comisura labial / sonrisa"),
    "AU45": ("Blink",          "parpadeo / cierre ocular"),
}

# Estado global del boton SALIR (para el mouse callback)
_abort = False
_btn_rect = (0, 0, 0, 0)   # x1, y1, x2, y2 — se actualiza cada frame


def mouse_callback(event, x, y, flags, param):
    """Detecta clic en el boton [SALIR]."""
    global _abort
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = _btn_rect
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            _abort = True


# ── Calculo de AUs ───────────────────────────────────────────────────────────

def compute_aus(gray, x, y, w, h):
    """Calcula Action Units aproximadas desde ROI facial con Haar Cascades."""
    roi = gray[y:y+h, x:x+w]

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))

    au4  = 0.0
    au7  = 0.0
    au12 = 0.0
    au45 = 0.0

    if len(eyes) >= 2:
        eyes     = sorted(eyes, key=lambda e: e[0])
        avg_area = np.mean([e[2] * e[3] for e in eyes[:2]])
        openness = avg_area / (w * h * 0.12 + 1e-6)

        au45 = float(np.clip(1.0 - openness,       0.0, 1.0))
        au7  = float(np.clip(0.6 - openness,       0.0, 1.0))
        au4  = float(np.clip(abs(eyes[0][1] - eyes[1][1]) / (h + 1e-6) * 5, 0.0, 1.0))
    else:
        # Ojos no detectados -> posible cierre fuerte
        au45, au7, au4 = 0.9, 0.7, 0.5

    mouth = roi[int(h * 0.65):, int(w * 0.2):int(w * 0.8)]
    if mouth.size > 0:
        _, thresh = cv2.threshold(mouth, 60, 255, cv2.THRESH_BINARY_INV)
        dark = thresh.sum() / (thresh.size * 255 + 1e-6)
        au12 = float(np.clip(1.0 - dark * 3, 0.0, 1.0))

    return {"AU4": au4, "AU7": au7, "AU12": au12, "AU45": au45}


# ── Dibujo en frame ───────────────────────────────────────────────────────────

def draw_face(frame, box, aus, conf):
    """Dibuja bbox, valores AU con descripcion y barra de nivel."""
    x1, y1, x2, y2 = box

    # Color segun nivel de AU45 (cierre ocular)
    if aus["AU45"] > 0.65:
        color = (50, 50, 230)     # rojo BGR
    elif aus["AU45"] > 0.35:
        color = (30, 158, 245)    # naranja/amarillo
    else:
        color = (80, 200, 60)     # verde

    # Bounding box de la cara
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Etiqueta de confianza encima del bbox
    cv2.putText(frame, f"Cara  conf:{conf:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Panel de AUs a la derecha del bbox
    px = x2 + 10
    py = y1
    row_h = 36

    for i, (au_key, (au_name, au_desc)) in enumerate(AU_INFO.items()):
        val = aus[au_key]
        ry  = py + i * row_h

        # Nombre + valor
        label = f"{au_key}  {val:.2f}  {au_name}"
        cv2.putText(frame, label,
                    (px, ry + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Descripcion en gris mas pequeno
        cv2.putText(frame, au_desc,
                    (px, ry + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        # Barra de nivel
        bar_w = 110
        filled = int(val * bar_w)
        bar_color = (
            (50, 50, 220)  if val > 0.65 else
            (30, 158, 245) if val > 0.35 else
            (80, 200, 60)
        )
        cv2.rectangle(frame, (px, ry + 27), (px + bar_w, ry + 31), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(frame, (px, ry + 27), (px + filled, ry + 31), bar_color, -1)

    return frame


def draw_legend(frame):
    """
    Panel fijo en esquina inferior izquierda explicando las AUs detectadas.
    Se muestra siempre, haya cara o no.
    """
    h, w = frame.shape[:2]
    lines = [
        "AUs reconocidas en este pipeline:",
        "AU4  Brow Lowerer  — ceno fruncido",
        "AU7  Lid Tightener — tension parpado",
        "AU12 Lip Corner    — comisura/sonrisa",
        "AU45 Blink         — parpadeo/cierre",
        "[MOCKUP] valores aproximados con Haar",
    ]
    pad    = 8
    lh     = 16
    box_h  = len(lines) * lh + pad * 2
    box_w  = 300
    box_y  = h - box_h - 10
    box_x  = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, text in enumerate(lines):
        color = (160, 160, 160) if i > 0 else (200, 200, 200)
        if i == len(lines) - 1:
            color = (100, 180, 100)   # nota mockup en verde tenue
        cv2.putText(frame, text,
                    (box_x + pad, box_y + pad + lh * i + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return frame


def draw_abort_button(frame):
    """
    Dibuja boton [SALIR] en esquina superior derecha.
    Actualiza _btn_rect para que el mouse callback lo detecte.
    """
    global _btn_rect
    h, w = frame.shape[:2]
    label  = "[ SALIR ]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    pad    = 8
    bx2    = w - 10
    by1    = 10
    bx1    = bx2 - tw - pad * 2
    by2    = by1 + th + pad * 2

    _btn_rect = (bx1, by1, bx2, by2)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 220), -1)
    cv2.putText(frame, label,
                (bx1 + pad, by2 - pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(frame, "o presiona Q",
                (bx1, by2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _abort

    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    model = YOLO(MODEL)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {source}")
        return

    win_name = "AU Detector — [MOCKUP]"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("[INFO] Corriendo — clic en [SALIR] o presiona Q para cerrar.")

    while True:
        if _abort:
            print("[INFO] Abortado por el usuario.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, conf=0.5, classes=[0], verbose=False)

        faces = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                aus  = compute_aus(gray, x1, y1, x2 - x1, y2 - y1)
                frame = draw_face(frame, (x1, y1, x2, y2), aus, conf)
                faces += 1
                print(f"AU4={aus['AU4']:.2f}  AU7={aus['AU7']:.2f}  "
                      f"AU12={aus['AU12']:.2f}  AU45={aus['AU45']:.2f}")

        if faces == 0:
            cv2.putText(frame, "Sin cara detectada",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (100, 100, 100), 2)

        frame = draw_legend(frame)
        frame = draw_abort_button(frame)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
