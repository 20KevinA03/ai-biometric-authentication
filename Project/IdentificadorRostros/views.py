# views.py
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse, Http404
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from .forms import RegistroEmpleadoForm, MarcarEntradaForm
from .models import Empleado, Asistencia
from .Services.systemRecognition import Camera, FACES_DIR
import os

# Anti-rebote (minutos entre marcas de la misma persona)
ANTI_REBOTE_MIN = 2

# Cámara única en memoria
_cam = None


def _get_cam():
    global _cam
    return _cam


def _kill_cam():
    """Libera y limpia la instancia global de cámara."""
    global _cam
    if _cam:
        try:
            _cam.stop()
        except Exception:
            pass
    _cam = None


def _set_cam(cam):
    """Apaga la cámara previa (si la hay) y asigna la nueva."""
    global _cam
    if _cam:
        try:
            _cam.stop()
        except Exception:
            pass
    _cam = cam


def menu(request):
    # Al entrar al menú cerramos cualquier cámara que haya quedado abierta
    _kill_cam()
    return render(request, "menu.html")


# ------------------ REGISTRO ------------------

@ensure_csrf_cookie
def registrar(request):
    # Cualquier GET de registro mata cámaras previas (por ejemplo, al volver desde la cámara)
    if request.method == "GET":
        _kill_cam()

    if request.method == "POST":
        form = RegistroEmpleadoForm(request.POST)
        if form.is_valid():
            nombre = form.cleaned_data["nombre"].strip()
            documento = form.cleaned_data["documento"]
            request.session["pending_reg"] = {
                "nombre": nombre,
                "documento": str(documento),
            }

            # preparar cámara en modo registro
            cam = Camera(mode="reg", documento=str(documento), nombre=nombre)
            _set_cam(cam)
            return render(request, "camera_BiometricR.html")
    else:
        form = RegistroEmpleadoForm()
    return render(request, "register.html", {"form": form})


def finalizar_registro(request):
    cam = _get_cam()
    if not cam or not cam.auth_ok:
        msg = cam.last_message if cam else "Autenticación aún no completada."
        return JsonResponse({"ok": False, "message": msg}, status=400)

    pending = request.session.get("pending_reg")
    if not pending:
        _kill_cam()
        return JsonResponse({"ok": False, "message": "No hay datos de registro en sesión."}, status=400)

    nombre = pending.get("nombre", "").strip()
    documento = pending.get("documento", "").strip()
    if not nombre or not documento:
        _kill_cam()
        return JsonResponse({"ok": False, "message": "Datos incompletos."}, status=400)

    # si ya existe, solo cerramos biometría
    if Empleado.objects.filter(documento=documento).exists():
        request.session.pop("pending_reg", None)
        _kill_cam()
        return JsonResponse({"ok": True, "message": "Documento ya registrado. Autenticación completada."})

    # crear empleado nuevo
    Empleado.objects.create(nombre=nombre, documento=documento)
    request.session.pop("pending_reg", None)
    _kill_cam()

    return JsonResponse({"ok": True})


# ------------------ LOGIN / MARCAR ENTRADA ------------------

@ensure_csrf_cookie
def ingresar(request):
    # Al llegar por GET, nos aseguramos de que no quede cámara previa viva
    if request.method == "GET":
        _kill_cam()

    if request.method == "POST":
        form = MarcarEntradaForm(request.POST)
        if form.is_valid():
            doc = form.cleaned_data["documento"].strip()
            nombre_optional = form.cleaned_data.get("nombre", "").strip()

            try:
                emp = Empleado.objects.get(documento=doc)
            except Empleado.DoesNotExist:
                return render(request, "login.html", {
                    "form": form,
                    "error": "Documento no encontrado o inactivo."
                })

            face_path = os.path.join(FACES_DIR, f"{doc}.png")
            if not os.path.exists(face_path):
                return render(request, "login.html", {
                    "form": form,
                    "error": "No hay rostro registrado para este documento."
                })

            request.session["pending_login"] = {"documento": doc, "nombre": emp.nombre}

            # Prepara cámara en modo login
            cam = Camera(mode="log", documento=doc, nombre=emp.nombre)
            _set_cam(cam)
            return render(request, "camera_BiometricL.html")
    else:
        form = MarcarEntradaForm()
    return render(request, "login.html", {"form": form})


def finalizar_ingreso(request):
    cam = _get_cam()
    if not cam or not cam.auth_ok:
        msg = cam.last_message if cam else "Validación pendiente."
        return JsonResponse({"ok": False, "message": msg}, status=400)

    pend = request.session.get("pending_login")
    if not pend:
        _kill_cam()
        return JsonResponse({"ok": False, "message": "No hay sesión de ingreso activa."}, status=400)

    doc = pend["documento"]
    try:
        emp = Empleado.objects.get(documento=doc)
    except Empleado.DoesNotExist:
        _kill_cam()
        return JsonResponse({"ok": False, "message": "Empleado no encontrado."}, status=404)

    hoy = timezone.localdate()
    hace = timezone.now() - timezone.timedelta(minutes=ANTI_REBOTE_MIN)
    ya = Asistencia.objects.filter(
        empleado=emp,
        cuando__date=hoy,
        cuando__gte=hace
    ).order_by("-cuando").first()

    if ya:
        request.session["ultima_marca"] = {
            "nombre": emp.nombre,
            "documento": emp.documento,
            "mensaje": "Ya existe una marcación reciente.",
            "cuando": ya.cuando.strftime("%Y-%m-%d %H:%M:%S"),
        }
        request.session.pop("pending_login", None)
        _kill_cam()
        return JsonResponse({"ok": True})

    # registra asistencia
    ahora = timezone.now()
    Asistencia.objects.create(empleado=emp, cuando=ahora)
    request.session["ultima_marca"] = {
        "nombre": emp.nombre,
        "documento": emp.documento,
        "mensaje": "Marcación de entrada registrada.",
        "cuando": ahora.strftime("%Y-%m-%d %H:%M:%S"),
    }
    request.session.pop("pending_login", None)
    _kill_cam()

    return JsonResponse({"ok": True})


# ------------------ STREAM / ESTADO / STOP ------------------

def video_feed(request):
    """img src con stream multipart; usa el modo actual del _cam ya inicializado por registrar/ingresar."""
    cam = _get_cam()
    if not cam:
        return HttpResponse(status=404)
    return StreamingHttpResponse(cam.frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def camera_status(request):
    cam = _get_cam()
    if not cam:
        return JsonResponse({"ok": False, "message": "Cámara no inicializada."})
    return JsonResponse({
        "ok": cam.auth_ok,
        "message": cam.last_message or "",
        "blinks": cam.conteo,
        "visualizando": cam.visualizando,
        "glass": cam.glass,
        "cap": cam.capHat
    })


def stop_camera(request):
    """Endpoint ligero para matar la cámara cuando el usuario sale de la página."""
    _kill_cam()
    return JsonResponse({"ok": True})


# imagen del rostro guardado para mostrar en success (no requiere static)
def face_image(request, documento):
    path = os.path.join(FACES_DIR, f"{documento}.png")
    if not os.path.exists(path):
        raise Http404()
    with open(path, "rb") as f:
        data = f.read()
    return HttpResponse(data, content_type="image/png")


def success(request):
    data = request.session.pop("ultima_marca", None)
    return render(request, "success.html", {"data": data})
