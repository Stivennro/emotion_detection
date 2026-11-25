"""Pruebas del asistente con un Gemini SIMULADO (no usan internet ni gastan cuota).

Ejecutar desde la raíz del proyecto:  python -m pytest tests -v
"""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors

from src.assistant import (FRIENDLY_ERRORS, MAX_MESSAGE_CHARS, GeminiAssistant, build_emotion_context,
                           message_text, to_gemini_history, validate_message)
from src.emotion_tracking import EmotionHistory

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FakeModels:
    """Imita client.models: devuelve respuestas o lanza errores según un guion por modelo."""

    def __init__(self, script):
        self.script = script       # {nombre_del_modelo: "texto" o código de error (int)}
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.script[model]
        if isinstance(outcome, int):
            raise errors.APIError(outcome, {"error": {"code": outcome, "message": "simulated", "status": "X"}})
        return SimpleNamespace(text=outcome)


def make_assistant(script):
    fake = SimpleNamespace(models=FakeModels(script))
    assistant = GeminiAssistant(api_key=None, model="main-model", fallback_model="backup-model", client=fake)
    return assistant, fake.models


def make_history(pattern, start=1_000_000):
    """Historial con una muestra por segundo; `pattern` es una lista de (emoción, segundos)."""
    history, second = EmotionHistory(CLASS_NAMES), 0
    for emotion, seconds in pattern:
        for _ in range(seconds):
            probs = np.full(7, 0.05)
            probs[CLASS_NAMES.index(emotion)] = 0.70
            history.add(probs, now=start + second)
            second += 1
    return history, start + second - 1


# ----------------------------------------------------------------------------- contexto
def test_context_without_samples():
    context = build_emotion_context(EmotionHistory(CLASS_NAMES), now=0)
    assert "no hay muestras" in context


def test_context_has_only_labels_and_percentages():
    history, last = make_history([("neutral", 40), ("happy", 20)])
    context = build_emotion_context(history, now=last + 1)
    assert "felicidad" in context and "neutral" in context
    assert "se está viendo ahora" in context
    assert "Últimos 30 s" in context and "Toda la sesión" in context
    assert "67 %" in context                     # 40 de 60 muestras neutral en la sesión
    assert "base64" not in context.lower()       # nunca imágenes


def test_context_warns_when_samples_are_old():
    history, last = make_history([("sad", 15)])
    context = build_emotion_context(history, now=last + 120)
    assert "AVISO" in context and "hace 120 s" in context


# ----------------------------------------------------------------------------- validación
@pytest.mark.parametrize("message", ["", "   ", None, "\n\t"])
def test_empty_messages_are_rejected(message):
    text, warning = validate_message(message)
    assert text is None and warning


def test_long_messages_are_rejected():
    text, warning = validate_message("a" * (MAX_MESSAGE_CHARS + 1))
    assert text is None and str(MAX_MESSAGE_CHARS) in warning


def test_message_is_cleaned():
    assert validate_message("  ¿Cómo   me\nveo?  ") == ("¿Cómo me veo?", None)


# ----------------------------------------------------------------------------- formato del chat
def test_message_text_accepts_gradio6_parts_and_strings():
    assert message_text([{"type": "text", "text": "hola"}]) == "hola"
    assert message_text("hola") == "hola"


def test_history_conversion_uses_gemini_roles_and_limit():
    chat = [{"role": "user" if i % 2 == 0 else "assistant", "content": [{"type": "text", "text": f"m{i}"}]}
            for i in range(12)]
    contents = to_gemini_history(chat, max_messages=4)
    assert [c["parts"][0]["text"] for c in contents] == ["m8", "m9", "m10", "m11"]
    assert {c["role"] for c in contents} == {"user", "model"}


# ----------------------------------------------------------------------------- llamadas a Gemini
def test_reply_sends_context_with_question():
    assistant, models = make_assistant({"main-model": "Te ves feliz."})
    reply = assistant.reply("¿Cómo me veo?", [], "DATOS DEL DETECTOR: felicidad 70 %")
    assert reply.ok and reply.text == "Te ves feliz." and reply.model == "main-model"
    sent = models.calls[0]["contents"][-1]["parts"][0]["text"]
    assert "DATOS DEL DETECTOR" in sent and "¿Cómo me veo?" in sent
    assert "No diagnostiques" not in sent        # las reglas van en la system instruction, no en el mensaje
    assert "diagnostiques" in models.calls[0]["config"].system_instruction


@pytest.mark.parametrize("code", [429, 503])
def test_fallback_model_is_used_when_main_is_busy(code):
    assistant, models = make_assistant({"main-model": code, "backup-model": "Respuesta del respaldo."})
    reply = assistant.reply("hola", [], "")
    assert reply.ok and reply.used_fallback and reply.model == "backup-model"
    assert [c["model"] for c in models.calls] == ["main-model", "backup-model"]


def test_friendly_message_when_all_models_fail():
    assistant, _ = make_assistant({"main-model": 503, "backup-model": 503})
    reply = assistant.reply("hola", [], "")
    assert not reply.ok and reply.text == FRIENDLY_ERRORS["busy"] and reply.error == "503"


def test_invalid_key_does_not_try_fallback():
    assistant, models = make_assistant({"main-model": 403, "backup-model": "no debería llamarse"})
    reply = assistant.reply("hola", [], "")
    assert not reply.ok and reply.text == FRIENDLY_ERRORS["auth"] and len(models.calls) == 1


def test_missing_key_gives_setup_message():
    assistant = GeminiAssistant(api_key=None)
    assert not assistant.available
    assert assistant.reply("hola").text == FRIENDLY_ERRORS["no_key"]


def test_stats_do_not_store_conversation_text():
    assistant, _ = make_assistant({"main-model": 503, "backup-model": "texto privado"})
    assistant.reply("pregunta privada", [], "")
    summary = assistant.performance_summary()
    assert summary["queries"] == 1 and summary["fallback_answers"] == 1
    assert "privad" not in str(assistant.stats) + str(summary)


def test_timeout_moves_to_fallback():
    class TimeoutModels(FakeModels):
        def generate_content(self, model, contents, config):
            if model == "main-model":
                self.calls.append({"model": model})
                raise TimeoutError("simulated read timeout")
            return super().generate_content(model, contents, config)

    fake = SimpleNamespace(models=TimeoutModels({"backup-model": "Respuesta rápida."}))
    assistant = GeminiAssistant(api_key=None, model="main-model", fallback_model="backup-model", client=fake)
    reply = assistant.reply("hola", [], "")
    assert reply.ok and reply.used_fallback and reply.text == "Respuesta rápida."


def test_client_has_timeout_and_no_internal_retries():
    from src.assistant import REQUEST_TIMEOUT_MS
    options = GeminiAssistant(api_key="fake-key").client._api_client._http_options
    assert options.timeout == REQUEST_TIMEOUT_MS
    assert options.retry_options.attempts == 1
