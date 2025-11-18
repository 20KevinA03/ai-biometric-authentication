🧠 Sistema de Reconocimiento Facial sin Accesorios
Registro y Marcación de Asistencia con Validación Biométrica en Tiempo Real

Python 3.11.9 · Django 5.2.7 · YOLOv8.3 · MediaPipe 0.10.14

📌 Descripción del Proyecto

Este proyecto implementa un sistema biométrico de registro y marcación de asistencia basado en visión por computador. El sistema permite:

✔ Registrar empleados capturando su rostro sin accesorios
✔ Validar identidad mediante reconocimiento facial
✔ Detectar gafas y gorras con YOLOv8
✔ Validar liveness mediante conteo de parpadeos con MediaPipe Face Mesh
✔ Ejecutar autenticación en tiempo real desde la cámara
✔ Prevenir suplantación y falsos positivos

Todo el flujo corre en una arquitectura web basada en Django, integrando modelos de visión por computador dentro de un entorno de Streaming HTTP.

🚀 Características Principales
🔍 1. Registro Biométrico

Captura automática del rostro

Requiere 3 parpadeos (prueba de vida)

Valida que el usuario no tenga gafas/gorras

Almacena la imagen y su encoding facial

🔐 2. Validación para Ingreso

Repite la prueba de vida

Detecta accesorios

Compara el rostro capturado con el registrado

Aplica umbral de similitud basado en dlib/face_recognition

🎥 3. Procesamiento en Tiempo Real

Videostream con StreamingHttpResponse

Detección cada frame:

Landmarks faciales (MediaPipe)

Detección de accesorios (YOLOv8)

Codificación facial (dlib)

⚙️ 4. Otros detalles técnicos

Manejo correcto del ciclo de vida de la cámara

Evita que la cámara quede en “limbo”

Apagado automático al cambiar de vista

Modelo de anti-rebote para asistencia

Guardado de rostros en disco

Encodings consistentes entre registro y login

🧩 Tecnologías Utilizadas
Componente	Versión	Descripción
Python	3.11.9	Lenguaje principal del backend
Django	5.2.7	Framework MVC para la arquitectura web
YOLOv8.3	Ultralytics	Detección de gafas y gorras
MediaPipe	0.10.14	Face Mesh y detección de parpadeos
dlib / face_recognition	últimas compatibles	Codificación facial y comparación
OpenCV	4.x	Manejo de video y transformaciones


📦 Instalación
1️⃣ Crear entorno virtual
python -m venv .venv


Activarlo:
.\.venv\Scripts\activate

2️⃣ Instalar dependencias
pip install django==5.2.7
pip install opencv-python
pip install mediapipe==0.10.14
pip install ultralytics==8.3.0
pip install face_recognition
pip install dlib
pip install numpy
