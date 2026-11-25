# Emotion Detection Assistant

Proyecto de la materia de Inteligencia Artificial (Ingeniería de Sistemas, FUP), por **Edwar Trujillo**.

Sistema de visión computacional que detecta rostros por webcam y clasifica su expresión en **7 emociones**
(enojo, asco, miedo, felicidad, neutral, tristeza y sorpresa) con una **red neuronal convolucional (CNN)** en
tiempo real. Incluye un **asistente conversacional** (Google Gemini) al que se le puede preguntar cómo se ve o cómo
se ha sentido la persona según el historial del detector.

**Funcionalidades**
- Detección de rostros con Haar Cascade (OpenCV) y clasificación con una CNN entrenada en FER-2013, con
  *test-time augmentation* (TTA).
- Aplicación web local (Gradio) con el video anotado, la emoción actual y su confianza, las probabilidades
  de las 7 emociones, la gráfica del historial y el chat con el asistente, todo en la misma pantalla.
- Suavizado temporal (media móvil de 10 cuadros) e historial en memoria (1 muestra por segundo).
- Asistente que solo recibe **etiquetas y porcentajes** (nunca imágenes), solo habla de emociones y no diagnostica.

## Resultados

### Modelo (conjunto de test de FER-2013, 7.178 imágenes, sin fuga de datos)

| Modelo | Accuracy | Top-2 | F1 macro |
|---|---|---|---|
| Base (réplica del repositorio de referencia) | 58,87 % | 77,38 % | 0,568 |
| Mejorado (BatchNorm, dropout, augmentation, class weights) | 64,00 % | 82,28 % | 0,605 |
| Mejorado v2 | 65,90 % | 83,37 % | 0,643 |
| **Mejorado v2 + TTA (modelo final, elegido por validación)** | **66,54 %** | **83,63 %** | **0,650** |

La precisión humana en FER-2013 ronda el 65 %. El 63,2 % que reporta el repositorio de referencia es optimista
porque usa el test como validación y no elimina las imágenes repetidas entre train y test (ver ADR-006).

### Tiempo real (CPU, Ryzen 5 7000, sin GPU)

| Medida | OpenCV directo (notebook 03) | Interfaz web Gradio (notebook 04) |
|---|---|---|
| FPS | 19,5 | 11,6 - 12,4 |
| Detección de rostro | 11,4 ms | 21-23 ms |
| Predicción (CNN + TTA, `tf.function`) | 19,3 ms | 33 ms |
| Parpadeo de la etiqueta con suavizado N = 10 | −66,7 % | −61 % a −71 % |

### Asistente (Gemini, plan gratuito)
16 de 16 consultas respondidas en una sesión de prueba, con una latencia media de **1,41 s** (máxima 1,85 s).

Las métricas completas están en `reports/metrics/` y las gráficas en `reports/figures/`.

## Arquitectura

```
 Navegador (cliente)                    Python local (servidor Gradio, 127.0.0.1)            Internet
 ┌──────────────────────┐  cuadro JPEG  ┌───────────────────────────────────────────┐
 │ webcam               │ ────────────▶ │ Haar Cascade → recorte 48x48 → CNN + TTA  │
 │ video anotado,       │ ◀──────────── │ → suavizado (N = 10) → historial (1/s)    │
 │ emoción, historial   │               │                                           │
 │ chat                 │ ◀───────────▶ │ build_emotion_context() (etiquetas y %) ──┼──▶ Google Gemini
 └──────────────────────┘               └───────────────────────────────────────────┘    (modelo principal
                                                                                          + respaldo)
```

## Estructura del repositorio

```
notebooks/
  01_data_exploration.ipynb        EDA de FER-2013: clases, píxeles, duplicados y fuga entre train y test
  02_model_training.ipynb          Entrenamiento y evaluación de los modelos (se ejecuta en Google Colab)
  03_realtime_detection.ipynb      Webcam con OpenCV: detección, predicción con TTA, suavizado e historial
  04_interface_and_assistant.ipynb Aplicación web con Gradio y asistente con Gemini
src/
  emotion_detection.py             EmotionDetector: carga el modelo y el Haar Cascade; detecta y clasifica
  emotion_tracking.py              EmotionSmoother (media móvil) y EmotionHistory (historial y resumen)
  assistant.py                     GeminiAssistant: contexto, reglas, validación y modelo de respaldo
tests/                             Pruebas automáticas (pytest) con Gemini simulado
models/                            Modelos .keras, class_names.json y selected_model.json
data/                              Dataset FER-2013 (no se sube a GitHub)
reports/figures/                   Gráficas (EDA, entrenamiento, evaluación, historial)
reports/metrics/                   Métricas en CSV y JSON
docs/adr.md                        Decisiones de arquitectura (ADR-001 a ADR-013)
CONTEXTO.md                        Estado actual del proyecto
.env.example                       Plantilla de la configuración del asistente
```

## Instalación (Windows con Anaconda)

Desde **Anaconda Prompt**, en la carpeta del proyecto:

```bat
conda create -n emotion-ai python=3.11 -y
conda activate emotion-ai
pip install -r requirements.txt
python -m ipykernel install --user --name emotion-ai --display-name "Python 3.11 (emotion-ai)"
```

- TensorFlow 2.21 soporta Python 3.10 a 3.13; en Windows nativo no usa GPU.
- Se usa **OpenCV 4.14** porque OpenCV 5 retiró el Haar Cascade del paquete principal.
  Si hace falta reinstalarlo: `pip install opencv-python==4.14.0.94`.

## Dataset

[FER-2013 en Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Se descomprime en `data/fer2013/`
de modo que queden `data/fer2013/train/` y `data/fer2013/test/`, con una carpeta por emoción.
Solo hace falta para los notebooks 01 y 02: el modelo entrenado ya está en `models/`.

## Configurar el asistente (Google Gemini)

1. Crea una API key gratuita en [Google AI Studio](https://aistudio.google.com/apikey).
2. Copia `.env.example` como `.env` en la raíz del proyecto y escribe tu clave:

   ```
   GEMINI_API_KEY=tu_clave
   GEMINI_MODEL=gemini-3.5-flash-lite
   GEMINI_FALLBACK_MODEL=gemini-3.1-flash-lite
   ```

`.env` está en `.gitignore`: nunca se sube al repositorio. Sin él, la aplicación funciona, pero el chat
muestra un aviso. Si Google retira alguno de los modelos, basta con cambiar su nombre en el `.env`.

## Uso

Abre Jupyter con el kernel **Python 3.11 (emotion-ai)** y ejecuta los notebooks en orden:

| Notebook | Dónde | Qué hace |
|---|---|---|
| `01_data_exploration` | Local | Análisis del dataset (necesita `data/`). |
| `02_model_training` | Google Colab (GPU) | Limpieza, entrenamiento y evaluación. Genera los modelos y las métricas. |
| `03_realtime_detection` | Local (webcam) | Demo con una ventana de OpenCV: `run_full_demo()`; se cierra con `q` o `Esc`. |
| `04_interface_and_assistant` | Local (webcam + internet) | **Aplicación completa.** |

**Aplicación completa (notebook 04):** ejecuta las Partes 1 a 3. Se abre `http://127.0.0.1:7860`; en
*Tu cámara* haz clic en **Click to Access Webcam**, permite el acceso y presiona **Record**. Luego pregúntale
al asistente (por ejemplo, *"¿cómo me veo ahora?"*). La Parte 4 guarda las métricas de la sesión. Para apagar
la aplicación, pon `STOP_SERVER = True` en la última celda y cierra la pestaña del navegador.

## Pruebas

```bat
python -m pytest tests -v
```

27 pruebas del suavizado, del historial y del asistente (contexto, validación, modelo de respaldo, *timeout*
y privacidad de las métricas). Usan un Gemini simulado, así que no necesitan internet ni gastan cuota.
Los notebooks incluyen además celdas de verificación (por ejemplo, que el modelo da las mismas
predicciones en Colab y en local).

## Privacidad y ética

- No se guardan imágenes ni video de la webcam: el video anotado viaja al navegador codificado en memoria y
  el historial solo contiene etiquetas y números.
- La aplicación solo es accesible desde el mismo PC (`127.0.0.1`).
- Al asistente solo se le envían etiquetas de emoción y porcentajes. La interfaz muestra exactamente qué se
  envía (*"Ver lo que se envía a Gemini"*). En el plan gratuito, Google puede usar ese contenido para mejorar
  sus productos; por eso nunca se envían imágenes ni datos personales.
- Una **expresión facial no equivale a la emoción real**: el sistema no hace diagnósticos y el asistente está
  instruido para no hacerlos. Los datos faciales son datos biométricos sensibles (Ley 1581 de 2012 en Colombia).

## Limitaciones

- Accuracy de 66,5 %: la emoción *asco* casi no se reconoce (solo 436 imágenes de entrenamiento) y con poca
  luz se confunden *neutral* y *enojo*.
- FER-2013 tiene etiquetas ruidosas y desbalance de clases, y puede no representar por igual a todas las
  poblaciones (posible sesgo).
- Haar Cascade falla con rostros girados o mal iluminados; se analiza solo el rostro más grande.
- El asistente depende de internet y de la disponibilidad del plan gratuito de Gemini (errores 429/503 y
  colas). Por eso tiene un modelo de respaldo y un límite de espera de 12 s.
- Es una demo local de un solo usuario: dos pestañas abiertas compartirían el mismo historial.

## Créditos

- Repositorio de referencia: [atulapra/Emotion-detection](https://github.com/atulapra/Emotion-detection).
- Dataset: FER-2013 (Goodfellow et al., 2013).
- Librerías: TensorFlow/Keras, OpenCV, Gradio, Google Gen AI SDK, NumPy, pandas, scikit-learn, Matplotlib.
