"""Detección de rostros y predicción de emociones (código de la Fase 3 hecho módulo).

Flujo por cuadro: imagen a color → grises → Haar Cascade (caja del rostro) → recorte 48x48
→ CNN (modelo mejorado v2) con TTA → 7 probabilidades.

Uso:
    from src.emotion_detection import EmotionDetector
    detector = EmotionDetector(PROJECT_ROOT / "models")
    boxes, probs, timings = detector.analyze_frame(frame_rgb, color_order="RGB")
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import keras

# --- Parámetros del detector Haar Cascade (elegidos en la Fase 3, parte 2.3) ---
SCALE_FACTOR = 1.1        # reduce la imagen 10 % en cada escala
MIN_NEIGHBORS = 5         # pide 5 detecciones superpuestas para aceptar una cara
MIN_FACE_SIZE = (60, 60)  # ignora caras menores a 60x60 píxeles (muy lejanas)
FACE_SIZE = 48            # tamaño de entrada del modelo (FER-2013)
MAX_FACES = 1             # solo el rostro más grande (evita falsos positivos del fondo)

# Etiquetas en español (sin tildes, porque cv2.putText no dibuja tildes) y colores en BGR
EMOTION_LABELS_ES = {
    "angry": "enojo", "disgust": "asco", "fear": "miedo", "happy": "felicidad",
    "neutral": "neutral", "sad": "tristeza", "surprise": "sorpresa",
}
EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128), "happy": (0, 215, 255),
    "neutral": (200, 200, 200), "sad": (255, 0, 0), "surprise": (0, 140, 255),
}

# Códigos de OpenCV para pasar a grises según el orden de los canales de la imagen
_TO_GRAY = {"BGR": cv2.COLOR_BGR2GRAY, "RGB": cv2.COLOR_RGB2GRAY}


def rgb_color(emotion):
    """Color de la emoción en RGB (0-255). OpenCV usa BGR; Gradio y el navegador usan RGB."""
    b, g, r = EMOTION_COLORS[emotion]
    return (r, g, b)


def hex_color(emotion):
    """Color de la emoción en formato web (#RRGGBB), para la interfaz y las gráficas."""
    return "#{:02x}{:02x}{:02x}".format(*rgb_color(emotion))


class EmotionDetector:
    """Carga el modelo y el Haar Cascade una sola vez y analiza cuadros de video."""

    def __init__(self, models_dir, max_faces=MAX_FACES):
        models_dir = Path(models_dir)

        # Orden de las clases tal como salen de la softmax (índice 0 = angry, ...)
        with open(models_dir / "class_names.json", encoding="utf-8") as f:
            self.class_names = json.load(f)
        # Modelo elegido por validación en la Fase 2 (ADR-007): mejorado v2 + TTA
        with open(models_dir / "selected_model.json", encoding="utf-8") as f:
            selected = json.load(f)

        self.model_path = models_dir / selected["model_file"]
        self.use_tta = bool(selected["use_tta"])
        self.max_faces = max_faces

        # compile=False: solo se predice, no hace falta el optimizador ni la pérdida
        self.model = keras.models.load_model(self.model_path, compile=False)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(cascade_path))
        if self.face_detector.empty():
            raise RuntimeError(
                "No se pudo cargar el Haar Cascade. Este proyecto usa OpenCV 4.x: "
                "pip install opencv-python==4.14.0.94"
            )

        # tf.function compila la red como un grafo: ~5 veces más rápido que model.predict()
        # para un solo rostro (Fase 3, parte 3). La firma fija evita recompilar en cada cuadro.
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, FACE_SIZE, FACE_SIZE, 1), dtype=tf.float32)])
        def forward_pass(batch):
            return self.model(batch, training=False)

        self._forward_pass = forward_pass
        self.predict(np.zeros((1, FACE_SIZE, FACE_SIZE), dtype=np.float32))  # calentamiento (compila)

    # ------------------------------------------------------------------ detección
    def detect_faces(self, gray):
        """Cajas (x, y, ancho, alto) de los rostros en grises, de la más grande a la más pequeña."""
        equalized = cv2.equalizeHist(gray)                     # más contraste, solo para detectar
        boxes = self.face_detector.detectMultiScale(
            equalized, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE_SIZE,
        )
        boxes = [tuple(int(v) for v in box) for box in boxes]  # numpy -> enteros de Python
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    @staticmethod
    def extract_face(gray, box, size=FACE_SIZE):
        """Recorta el rostro de la imagen en grises ORIGINAL y lo lleva a 48x48 (valores 0-255)."""
        x, y, w, h = box
        face = gray[y:y + h, x:x + w]
        return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)  # INTER_AREA: mejor al reducir

    # ------------------------------------------------------------------ predicción
    def predict(self, faces):
        """Probabilidades (N, 7) para rostros (N, 48, 48) o (N, 48, 48, 1) en grises 0-255.

        Con TTA se predice cada rostro y su espejo horizontal en un solo lote y se promedian.
        No se normaliza aquí: el modelo trae su propia capa Rescaling(1/255).
        """
        x = np.asarray(faces, dtype=np.float32)
        if x.ndim == 3:                                   # (N, 48, 48) -> (N, 48, 48, 1)
            x = x[..., np.newaxis]
        batch = np.concatenate([x, x[:, :, ::-1, :]]) if self.use_tta else x
        probs = self._forward_pass(tf.constant(batch)).numpy()
        if not self.use_tta:
            return probs
        n = len(x)
        return (probs[:n] + probs[n:]) / 2

    def analyze_frame(self, frame, color_order="BGR", max_faces="default"):
        """Detecta y clasifica los rostros de un cuadro a color.

        frame: matriz (alto, ancho, 3) en uint8.
        color_order: "BGR" si viene de OpenCV (cv2.VideoCapture) o "RGB" si viene de Gradio.
        Devuelve: cajas, probabilidades (N, 7) y los tiempos de cada paso en ms.
        """
        max_faces = self.max_faces if max_faces == "default" else max_faces
        gray = cv2.cvtColor(frame, _TO_GRAY[color_order])

        start = time.perf_counter()
        boxes = self.detect_faces(gray)
        if max_faces is not None:
            boxes = boxes[:max_faces]
        detection_ms = 1000 * (time.perf_counter() - start)

        start = time.perf_counter()
        if boxes:
            faces = np.stack([self.extract_face(gray, box) for box in boxes])
            probs = self.predict(faces)
        else:
            probs = np.empty((0, len(self.class_names)), dtype=np.float32)
        prediction_ms = 1000 * (time.perf_counter() - start)

        return boxes, probs, {"detection_ms": detection_ms, "prediction_ms": prediction_ms}


def draw_results(frame, boxes, probs, class_names, color_order="BGR"):
    """Copia del cuadro con un rectángulo por rostro y la emoción con su confianza."""
    output = frame.copy()
    for (x, y, w, h), face_probs in zip(boxes, probs):
        class_index = int(np.argmax(face_probs))
        emotion = class_names[class_index]
        color = EMOTION_COLORS[emotion] if color_order == "BGR" else rgb_color(emotion)
        label = f"{EMOTION_LABELS_ES[emotion]} {face_probs[class_index]:.0%}"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        # Fondo oscuro detrás del texto para que se lea con cualquier fondo
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_y = max(y - 10, text_h + 10)
        cv2.rectangle(output, (x, text_y - text_h - 6), (x + text_w + 6, text_y + 6), (0, 0, 0), -1)
        cv2.putText(output, label, (x + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return output
