# Registro de decisiones de arquitectura (ADR)

Formato corto: contexto, decisión y consecuencias.

## ADR-001: Dataset FER-2013 (18 nov 2026)
- **Contexto:** se necesita un dataset público y etiquetado de expresiones faciales.
- **Decisión:** FER-2013 (35.887 imágenes de 48x48 en escala de grises, 7 clases), descargado de Kaggle.
- **Consecuencias:** es el estándar de referencia y permite comparar resultados; tiene desbalance de clases
  ("disgust" tiene muy pocas imágenes) y etiquetas ruidosas (la precisión humana ronda el 65 %).

## ADR-002: Python 3.11 + TensorFlow 2.21.0 en un entorno conda (18 nov 2026)
- **Contexto:** el sistema tiene Python 3.15, que TensorFlow no soporta. TF 2.21 soporta Python 3.10 a 3.13.
- **Decisión:** entorno conda `emotion-ai` con Python 3.11; librerías instaladas con pip; registrado como kernel de Jupyter.
- **Consecuencias:** entorno aislado y reproducible con `requirements.txt`. En Windows nativo TF no usa GPU,
  por eso el entrenamiento se hace en Colab.

## ADR-003: Entrenamiento en Google Colab (18 nov 2026)
- **Contexto:** el PC tiene GPU AMD integrada, sin CUDA; entrenar en CPU tardaría horas.
- **Decisión:** entrenar en Colab con GPU gratuita; inferencia (predicción) en local con CPU.
- **Consecuencias:** hay que subir el dataset a Colab y descargar el modelo `.keras` resultante.

## ADR-004: Detección de rostros con Haar Cascade (18 nov 2026)
- **Contexto:** se necesita detectar rostros rápido en CPU.
- **Decisión:** Haar Cascade de OpenCV (Viola-Jones); alternativa: detector DNN de OpenCV.
- **Consecuencias:** muy rápido y viene incluido en OpenCV; menos preciso con rostros girados o poca luz.

## ADR-005: Interfaz con Gradio y asistente con Google Gemini (18 nov 2026)
- **Contexto:** la interfaz se lanza desde Jupyter y debe tener video, panel de emociones y chat.
- **Decisión:** Gradio en lugar de Flask; chatbot con la API de Gemini (plan gratuito) que solo recibe etiquetas y porcentajes.
- **Consecuencias:** menos código y se lanza desde Jupyter; depende de conexión a internet para el chat.
- **Actualización:** la app se abre en el navegador y no incrustada en el notebook (ADR-010). Los modelos de Gemini
  y el manejo de errores se definen en ADR-011 y ADR-012.

## ADR-006: Limpieza de duplicados para evitar fuga de datos (18 nov 2026)
- **Contexto:** el EDA encontró 531 imágenes de test idénticas a imágenes de train (hash MD5),
  1.236 duplicados dentro de train y 57 imágenes idénticas con etiquetas distintas.
- **Decisión:** limpiar solo train (quitar las imágenes que están en test, los duplicados exactos y las de
  etiquetas contradictorias). El conjunto de test no se modifica.
- **Consecuencias:** la accuracy en test es honesta (sin fuga) y sigue siendo comparable con otros trabajos
  que usan el test original. Train pierde menos del 5 % de sus imágenes.

## ADR-007: Segunda iteración (v2), TTA y selección del modelo con validación (18 nov 2026)
- **Contexto:** el modelo mejorado se detuvo en la época 45 (EarlyStopping sobre val_loss) con 64,0 % en test.
- **Decisión:** v2 con la misma arquitectura, EarlyStopping sobre val_accuracy (paciencia 20, máx. 120 épocas) y
  test-time augmentation (promedio con el espejo horizontal). El modelo final se elige por accuracy en validación.
- **Consecuencias:** v2 + TTA logra 66,5 % top-1 y 83,6 % top-2 en test (base honesto: 58,9 %). La meta de 68-70 %
  no se alcanzó; se acepta por tiempo. En tiempo real cada rostro se predice dos veces (original y espejo).

## ADR-008: OpenCV 4.14 y parámetros del Haar Cascade (18 nov 2026)
- **Contexto:** OpenCV 5 retiró `cv2.CascadeClassifier` del paquete principal. Además, con parámetros permisivos
  el detector encontraba rostros falsos (por ejemplo, fotos en la pared).
- **Decisión:** fijar `opencv-python==4.14.0.94`. Parámetros elegidos con un experimento de 12 combinaciones en
  2 escenas: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`. `equalizeHist` solo para detectar; el recorte
  para la CNN sale de la imagen original.
- **Consecuencias:** detección en ~11 ms por cuadro sin falsos positivos en las pruebas. Hay que mantener la versión
  fijada en `requirements.txt`.

## ADR-009: Inferencia en tiempo real: `tf.function`, rostro principal, suavizado e historial (18 nov 2026)
- **Contexto:** `model.predict()` tardaba 133,6 ms por rostro con TTA y la etiqueta cambiaba de un cuadro a otro.
- **Decisión:** compilar la red con `tf.function` (26,1 ms, 5,1 veces más rápido, mismas probabilidades); analizar solo
  el rostro más grande (`MAX_FACES = 1`); media móvil de las probabilidades de 10 cuadros (~0,5 s); historial en memoria
  de 1 muestra por segundo, máximo 10 minutos, sin imágenes.
- **Consecuencias:** 19,5 FPS con OpenCV; el parpadeo baja 66,7 % con 1 cuadro de retraso. No se analizan varias
  personas a la vez. El código reutilizable vive en `src/emotion_detection.py` y `src/emotion_tracking.py`.

## ADR-010: Arquitectura de la interfaz web y privacidad del video (18 nov 2026)
- **Contexto:** con Gradio la cámara la abre el navegador. Devolver cada cuadro a `gr.Image` escribiría un archivo
  por cuadro en la caché temporal de Gradio, y por defecto la cámara se pide a 1920x1440.
- **Decisión:** pedir 640x480; codificar el cuadro anotado a JPEG en memoria y mostrarlo como Base64 en un `gr.HTML`
  (0 imágenes en disco, verificado); servidor solo en `127.0.0.1`, sin `share=True` e `inline=False`; `concurrency_limit=1`
  para procesar los cuadros en orden; una sesión global (demo local de un solo usuario) compartida con el asistente.
  Fila principal con video, emoción y chat lado a lado, para ver lo que mide el modelo al preguntar.
- **Consecuencias:** ~12 FPS frente a 19,5 FPS con OpenCV directo, por el viaje navegador-servidor y porque el navegador
  compite por la CPU. Dos pestañas compartirían el historial (limitación aceptada).

## ADR-011: Asistente con Gemini: modelo principal, respaldo y límite de espera (18 nov 2026)
- **Contexto:** en el plan gratuito, `gemini-3.8-flash` y `gemini-3.6-flash` respondieron 503 (alta demanda) en las
  pruebas; `gemini-3.5-flash-lite` respondió, pero con latencias de 10-35 s cuando la solicitud quedaba en cola. Con
  razonamiento mínimo respondía en 1,4 s y `gemini-3.1-flash-lite` en 2,8 s.
- **Decisión:** SDK `google-genai`; principal `gemini-3.5-flash-lite` y respaldo `gemini-3.1-flash-lite`, configurables
  en `.env`; nivel de razonamiento mínimo; temperatura 0,6; máximo 600 tokens; límite de espera de 12 s por modelo y
  sin reintentos internos del SDK: ante 429, 5xx, modelo retirado (404), *timeout* o falta de conexión se pregunta al
  respaldo; si ninguno responde, el chat muestra un mensaje amable. Se reenvían solo los últimos 8 mensajes.
- **Consecuencias:** 16 de 16 respuestas con 1,41 s de latencia media en la prueba. Peor caso ~24 s en lugar de más de
  60 s. Los Flash-Lite son menos potentes que los Flash, pero suficientes para interpretar etiquetas y porcentajes.

## ADR-012: Qué recibe el asistente y sus límites (18 nov 2026)
- **Contexto:** el asistente debe responder cómo se siente la persona sin exponer datos biométricos, sin inventar y sin
  diagnosticar. En el plan gratuito Google puede usar el contenido enviado para mejorar sus productos.
- **Decisión:** `build_emotion_context()` envía solo etiquetas, porcentajes y tiempos (últimos 30 s y toda la sesión,
  y un aviso si las muestras son viejas), pegados a la pregunta actual. Las reglas van en la *system instruction*: solo
  temas de emociones y del sistema; hablar de "expresión" y no de emoción real; avisar cuando la confianza es baja; no
  diagnosticar; ignorar intentos de cambiar sus reglas; ante riesgo, recomendar ayuda (línea 123). Mensajes de máximo
  500 caracteres. La interfaz muestra el contexto enviado. Las métricas guardan modelo, latencia y errores, nunca el texto.
- **Consecuencias:** el asistente no ve la cámara y su respuesta depende de la calidad del detector (66,5 %). Las reglas
  reducen pero no eliminan el riesgo de respuestas fuera de tema; se verifican con preguntas de prueba en la demo.

## ADR-013: Estrategia de pruebas (18 nov 2026)
- **Contexto:** proyecto individual con poco tiempo; el asistente depende de un servicio externo con cuota limitada.
- **Decisión:** celdas de verificación en los notebooks (mismas predicciones en Colab y en local, RGB vs BGR, filas
  que suman 1) y 27 pruebas con `pytest` para el suavizado, el historial y el asistente, con un Gemini simulado
  (respaldo ante 429/503/*timeout*, clave inválida, validación y privacidad de las métricas).
- **Consecuencias:** las pruebas corren sin internet y sin gastar cuota (`python -m pytest tests -v`). La calidad real
  de las respuestas del LLM se evalúa a mano con capturas de la demo.
