"""Suavizado temporal e historial de emociones (código de la Fase 3, parte 4, hecho módulo).

- EmotionSmoother: media móvil de las probabilidades de los últimos N cuadros (N = 10).
- EmotionHistory: una muestra por segundo, solo en memoria (sin imágenes), máximo 10 minutos.
  Su método summary() es lo que recibirá el asistente en la Fase 5.
"""

import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

SMOOTHING_WINDOW = 10      # cuadros promediados (~0,5 s a 20 FPS)
MAX_MISSING_FRAMES = 10    # cuadros seguidos sin rostro antes de reiniciar el suavizado
HISTORY_INTERVAL_S = 1.0   # una muestra por segundo
HISTORY_MAX_ENTRIES = 600  # 10 minutos


class EmotionSmoother:
    """Media móvil de las probabilidades de los últimos `window` cuadros."""

    def __init__(self, window=SMOOTHING_WINDOW, max_missing=MAX_MISSING_FRAMES):
        self.window = window
        self.max_missing = max_missing
        self.buffer = deque(maxlen=window)   # con maxlen, al llenarse descarta el valor más viejo
        self.missing_frames = 0

    def update(self, probs):
        """Agrega las probabilidades del cuadro actual y devuelve el promedio de la ventana."""
        self.buffer.append(np.asarray(probs, dtype=np.float32))
        self.missing_frames = 0
        return np.mean(list(self.buffer), axis=0)

    def current(self):
        """Promedio actual de la ventana, o None si está vacía (sin rostro reciente)."""
        return np.mean(list(self.buffer), axis=0) if self.buffer else None

    def mark_missing(self):
        """Se llama cuando no hay rostro; si faltan demasiados cuadros seguidos, se reinicia."""
        self.missing_frames += 1
        if self.missing_frames > self.max_missing:
            self.reset()

    def reset(self):
        self.buffer.clear()
        self.missing_frames = 0


def count_label_changes(labels):
    """Cuántas veces cambia la etiqueta entre cuadros consecutivos (medida de "parpadeo")."""
    return int(sum(a != b for a, b in zip(labels[:-1], labels[1:])))


class EmotionHistory:
    """Historial en memoria de las emociones (solo etiquetas y probabilidades, sin imágenes)."""

    def __init__(self, class_names, interval_s=HISTORY_INTERVAL_S, max_entries=HISTORY_MAX_ENTRIES):
        self.class_names = list(class_names)
        self.interval_s = interval_s
        self.entries = deque(maxlen=max_entries)
        self._last_time = None

    def add(self, probs, now=None):
        """Guarda una muestra si ya pasó el intervalo. Devuelve True si se guardó."""
        now = time.time() if now is None else now
        if self._last_time is not None and now - self._last_time < self.interval_s:
            return False
        self._last_time = now
        index = int(np.argmax(probs))
        self.entries.append({
            "timestamp": now,
            "time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "emotion": self.class_names[index],
            "confidence": round(float(probs[index]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
        })
        return True

    def summary(self, last_seconds=None):
        """Resume el historial completo o solo los últimos `last_seconds` segundos."""
        entries = list(self.entries)
        if last_seconds is not None and entries:
            cutoff = entries[-1]["timestamp"] - last_seconds
            entries = [e for e in entries if e["timestamp"] > cutoff]
        if not entries:
            return {"samples": 0}
        counts = {c: 0 for c in self.class_names}
        for entry in entries:
            counts[entry["emotion"]] += 1
        dominant = max(counts, key=counts.get)
        return {
            "samples": len(entries),
            "seconds_covered": round(entries[-1]["timestamp"] - entries[0]["timestamp"] + self.interval_s, 1),
            "current_emotion": entries[-1]["emotion"],
            "current_confidence": entries[-1]["confidence"],
            "dominant_emotion": dominant,
            "distribution": {c: round(counts[c] / len(entries), 3) for c in self.class_names},
            "mean_confidence_dominant": round(float(np.mean(
                [e["confidence"] for e in entries if e["emotion"] == dominant])), 3),
        }

    def to_dataframe(self):
        """Tabla del historial (sin el detalle de las 7 probabilidades) para revisar o graficar."""
        columns = ["timestamp", "time", "emotion", "confidence"]
        rows = [{k: e[k] for k in columns} for e in list(self.entries)]
        return pd.DataFrame(rows, columns=columns)

    def clear(self):
        self.entries.clear()
        self._last_time = None
