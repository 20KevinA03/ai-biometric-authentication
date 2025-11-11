import math
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os

pathFace = 'C:/Users/kevin/OneDrive/Escritorio/IdentificadorDeRostros/Project/IdentificadorRostros/Services/Faces'

    
class Camera:

    #variables
    parpadeo = False
    conteo = 0
    muestra = 0
    step = 0
    

    #offset
    offsety = 40
    offsetx = 20

    #threshold
    confThreshold = 0.5

    def __init__(self, src=0, nombre=None, documento=None):

        self.cap = cv2.VideoCapture(src)
        self.nombre = nombre
        self.documento = documento
        self.auth_ok = False
        self.last_message = ""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        # FaceMesh una vez
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ==== estilos compatibles con cualquier versión ====
        try:
            self.mp_styles = mp.solutions.drawing_styles
            self.style_tesselation = self.mp_styles.get_default_face_mesh_tesselation_style()
            self.style_contours   = self.mp_styles.get_default_face_mesh_contours_style()
        except Exception:
            # fallback: usa DrawingSpec “manual”
            self.style_tesselation = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            self.style_contours    = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

        # puntos individuales:
        self.style_points = self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)

    def _resize_to_width(self, frame, width=1280):
        h, w = frame.shape[:2]
        if w == width:
            return frame
        scale = width / float(w)
        return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def frames(self):
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break
            frame_bgr = cv2.flip(frame_bgr, 1)
            frameSave = frame_bgr.copy()

            frame_bgr = self._resize_to_width(frame_bgr, 1280)
            #invertir la imagen para modo espejo
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = self.mesh.process(frame_rgb)

            px = []
            py = []
            lista = []

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    # malla completa
                    self.mp_draw.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=lm,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,               
                        connection_drawing_spec=self.style_tesselation
                    )
                    # contornos
                    self.mp_draw.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=lm,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.style_contours
                    )
                    # pintar también los puntos:
                    self.mp_draw.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=lm,
                        connections=[],  # sin conexiones
                        landmark_drawing_spec=self.style_points
                    )
                    #extract KeyPoints
                    for id, puntos in enumerate(lm.landmark):
                        #info img
                        al, an, c = frame_bgr.shape

                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        if len(lista) >= 478:
                            # ojo derecho 
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            # ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            #parietal derecho
                            x5, y5 = lista[139][1:]

                            #parietal izquierdo
                            x6, y6 = lista[368][1:]
                            
                            #ceja derecha
                            x7, y7 = lista[70][1:]

                            #ceja izquierda
                            x8, y8 = lista[300][1:]

                            #cv2.circle(frame_bgr,(x8,y8),2,(255,255,0),cv2.FILLED)
                            #cv2.circle(frame_bgr,(x6,y6),2,(255,0,0),cv2.FILLED)

                            faces = self.fd.process(frame_rgb) 
                            if faces.detections:
                                for face in faces.detections:
                                    score = face.score[0] if face.score else 0.0
                                    bbox = face.location_data.relative_bounding_box
                                    if score > 0.5:
                                        xi,yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi,yi, anc, alt = int(xi*an), int(yi*al), int(anc*an), int(alt*al)

                                        #offset x
                                        offsetan = (self.offsetx / 100 )* anc
                                        xi = int(xi- int(offsetan/2))
                                        anc = int(anc+offsetan)
                                        xf = xi + anc

                                        #offset y
                                        offsetal = (self.offsety / 100 )* alt
                                        yi = int(yi- offsetal)
                                        alt = int(alt+offsetal)
                                        cv2.rectangle(frame_bgr, (xi, yi, anc, alt), (255,255,255))
                                        yf = yi + alt

                                        if x7 > x5 and x8 < x6:
                                            #print("Viendo la camara")
                                            if longitud1 <= 10 and longitud2 <= 10 and self.parpadeo == False:
                                                self.conteo = self.conteo+1
                                                self.parpadeo = True
                                            elif longitud1 > 10 and longitud2 > 10 and self.parpadeo == True:
                                                self.parpadeo = False

                                            #print(f"parpadeos: {self.conteo}")
                                            
                                            if self.conteo >= 3  and not self.auth_ok:
                                                if longitud1 > 15 and longitud2 > 15:
                                                    #Cut
                                                    cut = frameSave[yi:yf, xi:xf]
                                                    #save Face
                                                    #print(f"Nombre {self.nombre}")
                                                    cv2.imwrite(f'{pathFace}/{self.documento}.png',cut)
                                                    self.auth_ok = True
                                                    self.last_message = "Autenticación biométrica exitosa."

                                        else:
                                            self.conteo = 0
            
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    def framesLog():
        print()
        images = []
        clases= []
        Lista = os.listdir(pathFace)

        #read face images
        for list in Lista:
            # read images
            imgdb = cv2.imread(f'{pathFace}/{list}')
            #save img db
            images.append(imgdb)
            #name img
            clases.append(os.path.splitext(list[0]))

        #faceCode = Code_Face(images)

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
    
    def set_empleado(self, nombre=None, documento=None):
        if nombre is not None:
            self.nombre = nombre
        if documento is not None:
            self.documento = documento
        # reset estado de autenticación cuando llega un nuevo usuario
        self.conteo = 0
        self.parpadeo = False
        self.auth_ok = False
        self.last_message = ""

    def Code_Face(images):
        listacod = []
        # Iteramos
        for img in images:
            # Correccion de color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Codificamos la imagen
            cod = fr.face_encodings(img)[0]
            # Almacenamos
            listacod.append(cod)

        return listacod
