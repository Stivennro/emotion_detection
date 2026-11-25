# CONTEXTO del proyecto

Archivo de continuidad entre sesiones. Se actualiza al cerrar cada fase.

## Stack
Python 3.11 (conda `emotion-ai`), Jupyter Notebook, Google Colab (entrenamiento con GPU),
TensorFlow/Keras 2.21.0 (Keras 3.15.1 local), OpenCV 4.14, NumPy 2.4, pandas, scikit-learn, Matplotlib/Seaborn,
Gradio 6.32.0 (interfaz), google-genai 3.6.1 + python-dotenv (asistente con Google Gemini), pytest (pruebas).
Versiones exactas en `requirements.txt`.

## Decisiones vigentes
Ver `docs/adr.md` (ADR-001 a ADR-013).

## Estado
- [x] Fase 1: entorno, repo, dataset y EDA (`notebooks/01_data_exploration.ipynb`) (18 nov)
  - 35.887 imágenes; desbalance 16,5 a 1; fuga detectada (531 imágenes de test en train) → ADR-006
- [x] Fase 2: entrenamiento en Colab (`notebooks/02_model_training.ipynb`) (18 nov)
  - Train limpio 24.220 / validación 2.692 / test 7.178
  - Base 58,9 % · Mejorado 64,0 % · v2 65,9 % · **v2 + TTA 66,5 % (top-2 83,6 %)**
  - Modelo final: `models/improved_model_v2.keras` con TTA (`models/selected_model.json`), elegido por validación
- [x] Fase 3: tiempo real con OpenCV (`notebooks/03_realtime_detection.ipynb`) (18 nov)
  - Haar Cascade 1.1/5/(60,60) (ADR-008); `tf.function` 5,1 veces más rápido; solo el rostro principal
  - 19,5 FPS; suavizado N = 10 → −66,7 % de parpadeo; historial 1 muestra/s en memoria (ADR-009)
- [x] Fase 4: interfaz con Gradio (`notebooks/04_interface_and_assistant.ipynb`, Partes 1-4) (18 nov)
  - Código reutilizable en `src/emotion_detection.py` y `src/emotion_tracking.py`
  - 640x480, video anotado en memoria (0 imágenes en disco), solo `127.0.0.1` (ADR-010)
  - 11,6-12,4 FPS; parpadeo −61 % a −71 % (`reports/metrics/gradio_performance.json`)
- [x] Fase 5: asistente con Gemini, pruebas y documentación (18 nov)
  - `src/assistant.py`: contexto con etiquetas y % (últimos 30 s y sesión), reglas, validación (máx. 500 caracteres),
    principal `gemini-3.5-flash-lite` → respaldo `gemini-3.1-flash-lite`, *timeout* 12 s (ADR-011, ADR-012)
  - Chat en la fila principal de la interfaz, junto al video y la emoción; preguntas rápidas; "Ver lo que se envía a Gemini"
  - 16/16 respuestas, latencia media 1,41 s (`reports/metrics/assistant_performance.json`)
  - `tests/`: 27 pruebas con pytest y Gemini simulado (ADR-013); README final
- [ ] Fase 6: reporte IEEE y diapositivas
- [ ] Fase 7: video y entrega (24 nov, en la noche)

## Cómo ejecutar
1. `conda activate emotion-ai` y crear `.env` a partir de `.env.example` (API key de Google AI Studio).
2. Notebook 04, Partes 1-3 → `http://127.0.0.1:7860` → *Click to Access Webcam* → *Record*.
3. Pruebas: `python -m pytest tests -v`.

## Próximos pasos
Fase 6: reporte en formato IEEE (introducción, trabajos relacionados, dataset y limpieza, modelos, tiempo real,
interfaz y asistente, resultados, limitaciones y ética, conclusiones) y diapositivas para el video.
