import os
import math
import cv2
import numpy as np
import mediapipe as mp
import dlib
import face_recognition as fr
import face_recognition_models

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False


# === Rutas dentro de la app ===
APP_DIR = os.path.dirname(__file__)
FACES_DIR = os.path.join(APP_DIR, "Faces")
MODELS_DIR = os.path.join(APP_DIR, "Models")
os.makedirs(FACES_DIR, exist_ok=True)


# === Carga perezosa de modelos YOLO (gafas / gorras) ===
_model_glass = None
_model_cap = None

# === dlib: detector / predictor / encoder (para evitar bugs de face_recognition.face_encodings) ===
_dlib_detector = dlib.get_frontal_face_detector()
_dlib_shape_predictor = dlib.shape_predictor(
    face_recognition_models.pose_predictor_model_location()
)
_dlib_face_encoder = dlib.face_recognition_model_v1(
    face_recognition_models.face_recognition_model_location()
)


def _encode_face_dlib(image_rgb):

    dets = _dlib_detector(image_rgb, 1)
    if not dets:
        return None

    shapes = dlib.full_object_detections()
    for d in dets:
        shapes.append(_dlib_shape_predictor(image_rgb, d))

    # Usa la sobrecarga (img, faces: full_object_detections)
    descriptors = _dlib_face_encoder.compute_face_descriptor(image_rgb, shapes)
    if len(descriptors) == 0:
        return None

    return np.array(descriptors[0])


def _get_models():
    global _model_glass, _model_cap

    if not _HAS_YOLO:
        return None, None

    if _model_glass is None:
        gafas_path = os.path.join(MODELS_DIR, "Gafas.pt")
        if os.path.exists(gafas_path):
            _model_glass = YOLO(gafas_path)

    if _model_cap is None:
        gorras_path = os.path.join(MODELS_DIR, "Gorros.pt")
        if os.path.exists(gorras_path):
            _model_cap = YOLO(gorras_path)

    return _model_glass, _model_cap


def _code_face(images_bgr):
    encs = []
    for img in images_bgr:
        if img is None or img.size == 0:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        enc = _encode_face_dlib(rgb)
        if enc is not None:
            encs.append(enc)
    return encs



class Camera:
    def __init__(self, src=0, mode="reg", nombre=None, documento=None):
        # === Cámara ===
        # Resolución más baja para mejor rendimiento (suficiente para biometría)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # === Datos de usuario / modo ===
        self.mode = mode  # 'reg' o 'log'
        self.nombre = nombre
        self.documento = documento

        # === Estado general ===
        self.visualizando = False
        self.parpadeo = False
        self.conteo = 0              # contador de parpadeos
        self.auth_ok = False         # autenticación completada
        self.last_message = ""       # mensaje que expones en Django
        self.glass = False           # detectó gafas
        self.capHat = False          # detectó gorra/sombrero
        self.glass_box = None
        self.cap_box = None
        self.offsety = 30            # márgenes de recorte para el rostro
        self.offsetx = 20
        self.frame_index = 0         # contar frames para no correr YOLO en todos

        # === MediaPipe: Face Mesh y Face Detection ===
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # === YOLO (si está disponible) ===
        self.model_glass, self.model_cap = _get_models()

        # === Para login: precargar embedding de referencia si existe ===
        self.ref_encoding = None
        self.ref_path = os.path.join(
            FACES_DIR, f"{self.documento}.png"
        ) if self.documento else None

        if self.mode == "log" and self.ref_path and os.path.exists(self.ref_path):
            ref = cv2.imread(self.ref_path)
            encs = _code_face([ref])
            if encs:
                self.ref_encoding = encs[0]

    # === Utilidades internas ===

    def _resize_to_width(self, frame, width=640):
        """Normaliza ancho para reducir carga (por defecto 640px)."""
        h, w = frame.shape[:2]
        if w == width:
            return frame
        new_h = int(h * (width / w))
        return cv2.resize(frame, (width, new_h))

    def stop(self):
        """Libera la cámara."""
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    # === YOLO: detectar gafas / gorra ===
    def _object_detection(self, frame_bgr):
        self.glass = False
        self.capHat = False
        self.glass_box = None
        self.cap_box = None
        if not (self.model_cap or self.model_glass):
            return

        try:
            h, w = frame_bgr.shape[:2]
            scale = 0.5
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

            # (modelo, atributo_bool, atributo_box, clase_objetivo, umbral_conf)
            detecciones = [
                (self.model_cap, "capHat", "cap_box", 1, 0.5),   # gorra/sombrero
                (self.model_glass, "glass", "glass_box", 0, 0.5)  # gafas
            ]

            for model, flag_attr, box_attr, target_cls, conf_thr in detecciones:
                if model is None:
                    continue

                # imgsz más pequeño para acelerar (320)
                res = model(small, imgsz=320)[0]

                for box in res.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_thr:
                        continue

                    cls_idx = int(box.cls[0])
                    if cls_idx != target_cls:
                        continue

                    # coordenadas en la imagen pequeña
                    x1s, y1s, x2s, y2s = box.xyxy[0].tolist()

                    # re-escalar a la imagen original
                    x1 = int(x1s / scale)
                    y1 = int(y1s / scale)
                    x2 = int(x2s / scale)
                    y2 = int(y2s / scale)

                    setattr(self, flag_attr, True)
                    setattr(self, box_attr, (x1, y1, x2, y2))
                    break  # con uno por frame es suficiente

        except Exception:
            self.glass = False
            self.capHat = False
            self.glass_box = None
            self.cap_box = None

    # === REGISTRO: guarda rostro al completar parpadeos ===

    def _pipeline_registro(self, frame_bgr, frame_rgb, frame_save):

        results = self.mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            self.last_message = "Acomódate frente a la cámara."
            return frame_bgr

        for lm in results.multi_face_landmarks:
            h, w = frame_bgr.shape[:2]
            pts = [[i, int(p.x * w), int(p.y * h)] for i, p in enumerate(lm.landmark)]
            if len(pts) < 387:
                continue

            # ojos
            x1, y1 = pts[145][1:]; x2, y2 = pts[159][1:]
            x3, y3 = pts[374][1:]; x4, y4 = pts[386][1:]
            L1 = math.hypot(x2 - x1, y2 - y1)
            L2 = math.hypot(x4 - x3, y4 - y3)

            # referencias laterales (parietal / cejas)
            x5, y5 = pts[139][1:]; x6, y6 = pts[368][1:]
            x7, y7 = pts[70][1:];  x8, y8 = pts[300][1:]

            faces = self.fd.process(frame_rgb)
            if not faces.detections:
                self.last_message = "No se detecta rostro."
                return frame_bgr

            for f in faces.detections:
                score = f.score[0] if f.score else 0
                if score < 0.5:
                    continue

                bbox = f.location_data.relative_bounding_box
                xi = int(bbox.xmin * w)
                yi = int(bbox.ymin * h)
                wc = int(bbox.width * w)
                hc = int(bbox.height * h)

                # ampliamos bbox con offsets
                xi = max(0, int(xi - (self.offsetx / 100) * wc / 2))
                wc = int(wc + (self.offsetx / 100) * wc)
                yi = max(0, int(yi - (self.offsety / 100) * hc))
                hc = int(hc + (self.offsety / 100) * hc)

                xf = min(w, xi + wc)
                yf = min(h, yi + hc)

                # bloqueo por gafas / gorra
                if self.capHat:
                    self.last_message = "Quita la gorra/sombrero para continuar."
                    if self.cap_box:
                        cx1, cy1, cx2, cy2 = self.cap_box
                        cv2.rectangle(frame_bgr, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (0, 0, 255), 2)
                    return frame_bgr

                if self.glass:
                    self.last_message = "Quita las gafas para continuar."
                    if self.glass_box:
                        gx1, gy1, gx2, gy2 = self.glass_box
                        cv2.rectangle(frame_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (0, 0, 255), 2)
                    return frame_bgr

                if not (x7 > x5 and x8 < x6):
                    self.last_message = "Mira de frente a la cámara."
                    self.conteo = 0
                    return frame_bgr

                self.last_message = "Parpadea 3 veces…"
                cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (255, 255, 255), 1)

                # detección de parpadeo
                if L1 <= 10 and L2 <= 10 and not self.parpadeo:
                    self.conteo += 1
                    self.parpadeo = True
                elif L1 > 10 and L2 > 10 and self.parpadeo:
                    self.parpadeo = False

                # al menos 3 parpadeos + ojos abiertos -> guardar rostro
                if self.conteo >= 3 and not self.auth_ok and (L1 > 12 and L2 > 12):
                    crop = frame_save[yi:yf, xi:xf]
                    try:
                        out_path = os.path.join(FACES_DIR, f"{self.documento}.png")
                        cv2.imwrite(out_path, crop)
                        self.auth_ok = True
                        self.last_message = "Autenticación biométrica exitosa."
                    except Exception as e:
                        self.last_message = f"Error al guardar el rostro: {e}"

        return frame_bgr

    # === LOGIN: compara contra embedding guardado ===

    def _pipeline_login(self, frame_bgr, frame_rgb):
        if self.ref_encoding is None:
            self.last_message = "No hay rostro registrado para este documento."
            return frame_bgr

        results = self.mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            self.last_message = "Acomódate frente a la cámara."
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        for lm in results.multi_face_landmarks:
            pts = [[i, int(p.x * w), int(p.y * h)] for i, p in enumerate(lm.landmark)]
            if len(pts) < 387:
                continue

            # ojos
            x1, y1 = pts[145][1:]; x2, y2 = pts[159][1:]
            x3, y3 = pts[374][1:]; x4, y4 = pts[386][1:]
            L1 = math.hypot(x2 - x1, y2 - y1)
            L2 = math.hypot(x4 - x3, y4 - y3)

            # referencias laterales (para verificar que mire al frente)
            x5, y5 = pts[139][1:]; x6, y6 = pts[368][1:]
            x7, y7 = pts[70][1:];  x8, y8 = pts[300][1:]

            # ===== bbox del rostro con FaceDetection (para dibujar recuadro) =====
            faces = self.fd.process(frame_rgb)
            if not faces.detections:
                self.last_message = "No se detecta rostro."
                return frame_bgr

            f = faces.detections[0]
            score = f.score[0] if f.score else 0
            if score < 0.5:
                self.last_message = "Rostro con baja confianza."
                return frame_bgr

            bbox = f.location_data.relative_bounding_box
            xi = int(bbox.xmin * w)
            yi = int(bbox.ymin * h)
            wc = int(bbox.width * w)
            hc = int(bbox.height * h)

            # ampliamos bbox con offsets
            xi = max(0, int(xi - (self.offsetx / 100) * wc / 2))
            wc = int(wc + (self.offsetx / 100) * wc)
            yi = max(0, int(yi - (self.offsety / 100) * hc))
            hc = int(hc + (self.offsety / 100) * hc)

            xf = min(w, xi + wc)
            yf = min(h, yi + hc)

            # ===== bloqueo por gafas / gorra con recuadro =====
            if self.glass:
                self.last_message = "Quita las gafas para continuar."
                if self.glass_box:
                    gx1, gy1, gx2, gy2 = self.glass_box
                    cv2.rectangle(frame_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (0, 0, 255), 2)
                return frame_bgr

            if self.capHat:
                self.last_message = "Quita la gorra/sombrero para continuar."
                if self.cap_box:
                    cx1, cy1, cx2, cy2 = self.cap_box
                    cv2.rectangle(frame_bgr, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (0, 0, 255), 2)
                return frame_bgr

            # Recuadro del rostro
            cv2.rectangle(frame_bgr, (xi, yi), (xf, yf), (255, 255, 255), 1)

            # ===== Verificar que mire al frente =====
            if not (x7 > x5 and x8 < x6):
                self.last_message = "Mira de frente a la cámara."
                self.conteo = 0
                return frame_bgr

            # ===== Parpadeos (liveness) =====
            if L1 <= 10 and L2 <= 10 and not self.parpadeo:
                self.conteo += 1
                self.parpadeo = True
            elif L1 > 10 and L2 > 10 and self.parpadeo:
                self.parpadeo = False

            if self.conteo < 3:
                self.last_message = f"Parpadea 3 veces para validar. Parpadeos: {self.conteo}"
                return frame_bgr

            if not (L1 > 14 and L2 > 14):
                self.last_message = "Mantén los ojos abiertos después de parpadear."
                return frame_bgr

            # ===== Reconocimiento facial (solo si pasó todo lo anterior) =====
            faces_loc = fr.face_locations(frame_rgb)
            encs = fr.face_encodings(frame_rgb, faces_loc)

            if not encs:
                self.last_message = "No se pudo codificar el rostro."
                return frame_bgr

            enc = encs[0]
            dist = fr.face_distance([self.ref_encoding], enc)[0]

            self.last_message = f"Similitud: {round(1 - float(dist), 3)}"

            if dist <= 0.50:
                self.auth_ok = True
                self.last_message = "Validación biométrica correcta."
            else:
                self.last_message += " — Rostro no coincide con el registrado."

            # solo procesamos el primer rostro
            return frame_bgr

        return frame_bgr

    # === Generador de frames para Django ===

    def frames(self):
        """Generador que se usa en el StreamingHttpResponse."""
        try:
            while True:
                if self.auth_ok:
                    break

                if self.cap is None or not self.cap.isOpened():
                    self.visualizando = False
                    break

                ok, frame_bgr = self.cap.read()
                if not ok:
                    self.visualizando = False
                    break

                self.visualizando = True
                self.frame_index += 1

                # espejo horizontal
                frame_bgr = cv2.flip(frame_bgr, 1)

                # copia sin dibujos para guardar rostro limpio
                frame_save = frame_bgr.copy()

                # normalizamos ancho
                frame_bgr = self._resize_to_width(frame_bgr)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # detección de accesorios solo cada 3 frames para mejorar rendimiento
                if self.frame_index % 3 == 0:
                    self._object_detection(frame_bgr)

                # flujo según modo
                if self.mode == "reg":
                    frame_bgr = self._pipeline_registro(frame_bgr, frame_rgb, frame_save)
                else:
                    frame_bgr = self._pipeline_login(frame_bgr, frame_rgb)

                # HUD: contador de parpadeos y estado cámara
                cv2.putText(
                    frame_bgr,
                    f"Parpadeos: {int(self.conteo)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                if self.visualizando:
                    cv2.putText(
                        frame_bgr,
                        "Cámara: Activa",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                # comprimimos el JPEG con calidad 80 (buen balance calidad/rendimiento)
                ok2, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok2:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
        finally:
            # Pase lo que pase, se libera la cámara al cerrar el generador
            self.stop()
