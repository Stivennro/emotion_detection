"""Pruebas del suavizado y del historial (src/emotion_tracking.py). No usan cámara ni modelo.

Ejecutar desde la raíz del proyecto:  python -m pytest tests -v
"""

import numpy as np

from src.emotion_tracking import EmotionHistory, EmotionSmoother, count_label_changes

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def one_hot(index):
    probs = np.zeros(7, dtype=np.float32)
    probs[index] = 1.0
    return probs


def test_smoother_is_the_mean_of_the_last_n_frames():
    smoother = EmotionSmoother(window=3)
    for index in (0, 0, 3, 3):                  # el primer 0 sale de la ventana
        result = smoother.update(one_hot(index))
    np.testing.assert_allclose(result[[0, 3]], [1 / 3, 2 / 3], atol=1e-6)
    assert np.isclose(result.sum(), 1.0)


def test_smoother_resets_after_too_many_missing_frames():
    smoother = EmotionSmoother(window=3, max_missing=2)
    smoother.update(one_hot(3))
    smoother.mark_missing()
    smoother.mark_missing()
    assert smoother.current() is not None       # pérdida breve: se mantiene
    smoother.mark_missing()
    assert smoother.current() is None           # pérdida larga: se reinicia


def test_count_label_changes():
    assert count_label_changes([1, 1, 2, 2, 1]) == 2
    assert count_label_changes([]) == 0


def test_history_keeps_one_sample_per_second():
    history = EmotionHistory(CLASS_NAMES)
    assert history.add(one_hot(3), now=100.0)
    assert not history.add(one_hot(3), now=100.5)   # menos de 1 s después: se ignora
    assert history.add(one_hot(4), now=101.0)
    assert len(history.entries) == 2


def test_history_summary_window_and_distribution():
    history = EmotionHistory(CLASS_NAMES)
    for second in range(60):
        history.add(one_hot(4 if second < 40 else 3), now=1000 + second)
    full, recent = history.summary(), history.summary(last_seconds=30)
    assert full["samples"] == 60 and full["dominant_emotion"] == "neutral"
    assert full["distribution"]["neutral"] == round(40 / 60, 3)
    assert recent["samples"] == 30 and recent["dominant_emotion"] == "happy"
    assert recent["current_emotion"] == "happy"


def test_history_has_no_images_and_respects_max_entries():
    history = EmotionHistory(CLASS_NAMES, max_entries=5)
    for second in range(10):
        history.add(one_hot(0), now=second)
    assert len(history.entries) == 5
    assert set(history.entries[0]) == {"timestamp", "time", "emotion", "confidence", "probabilities"}


def test_empty_summary():
    assert EmotionHistory(CLASS_NAMES).summary() == {"samples": 0}
