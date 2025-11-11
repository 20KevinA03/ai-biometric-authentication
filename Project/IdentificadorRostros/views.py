from django.shortcuts import render, redirect
from django.db import IntegrityError
from django.utils import timezone
from django.http import StreamingHttpResponse, HttpResponse,JsonResponse
from .forms import RegistroEmpleadoForm, MarcarEntradaForm
from .models import Empleado, Asistencia
from .Services.systemRecognition import Camera
from django.views.decorators.csrf import ensure_csrf_cookie

# Anti-rebote: evita 2 marcas en X minutos
ANTI_REBOTE_MIN = 2

def menu(request):
    return render(request, "menu.html")


_cam = None  # simple para no abrir varias veces

def _get_cam():
    global _cam
    if _cam is None:
        _cam = Camera(src=0)
    else:
        # si por cualquier razón la cap está cerrada: re-crea
        try:
            if not _cam.cap or not _cam.cap.isOpened():
                _cam = Camera(src=0)
        except Exception:
            _cam = Camera(src=0)
    return _cam

def video_feed(request):
    cam = _get_cam()
    return StreamingHttpResponse(
        cam.frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@ensure_csrf_cookie
def camera_page(request):
    return render(request, "camera_BiometricR.html")

@ensure_csrf_cookie
def cameraL_page(request):
    return render(request, "camera_BiometricL.html")

def camera_status(request):
    cam = _get_cam()
    data = {
        "ok": cam.auth_ok,
        "message": cam.last_message or "",
    }
    return JsonResponse(data)

def finalizar_registro(request):
    #Crea el Empleado solo si la autenticación fue exitosa.
    cam = _get_cam()
    if not cam.auth_ok:
        return JsonResponse({"ok": False, "message": "Autenticación aún no completada."}, status=400)

    pending = request.session.get("pending_reg")
    if not pending:
        return JsonResponse({"ok": False, "message": "No hay datos de registro en sesión."}, status=400)

    nombre = pending.get("nombre", "").strip()
    documento = pending.get("documento", "").strip()

    if not nombre or not documento:
        return JsonResponse({"ok": False, "message": "Datos incompletos."}, status=400)

    # Validación de duplicados
    if Empleado.objects.filter(documento=documento).exists():
        request.session.pop("pending_reg", None)
        cam.release()
        cam = None
        return JsonResponse({"ok": True, "message": "Documento ya registrado. Autenticación completada."})

    # Crear empleado
    emp = Empleado.objects.create(nombre=nombre, documento=documento)
    cam.set_empleado(nombre=nombre, documento=documento)

    # Limpiar sesión y cerrar cámara
    request.session.pop("pending_reg", None)
    cam.release()
    cam = None
    return JsonResponse({"ok": True, "message": "Usuario registrado correctamente"})

def registrar(request):
    if request.method == "POST":
        form = RegistroEmpleadoForm(request.POST)
        if form.is_valid():
            nombre = form.cleaned_data["nombre"].strip()
            documento = form.cleaned_data["documento"].strip()

            # Guardar en sesión y abrir cámara con estos datos
            request.session["pending_reg"] = {"nombre": nombre, "documento": documento}

            cam = _get_cam()
            cam.set_empleado(nombre=nombre, documento=documento)

            return redirect("camera_page")
    else:
        form = RegistroEmpleadoForm()
    return render(request, "register.html", {"form": form})

def ingresar(request):
    if request.method == "POST":
        form = MarcarEntradaForm(request.POST)
        if form.is_valid():
            doc = form.cleaned_data["documento"].strip()
            nombre_opt = (form.cleaned_data.get("nombre") or "").strip()

            try:
                emp = Empleado.objects.get(documento=doc, activo=True)
            except Empleado.DoesNotExist:
                form.add_error("documento", "Documento no encontrado o inactivo.")
                return render(request, "login.html", {"form": form})

            # Validación opcional del nombre si lo escriben
            if nombre_opt and emp.nombre.strip().lower() != nombre_opt.lower():
                form.add_error("nombre", "El nombre no coincide con el documento.")
                return render(request, "login.html", {"form": form})

            hoy = timezone.localdate()
            ahora = timezone.now()

            # Anti-rebote por tiempo
            ultima = emp.asistencias.first()
            if ultima and (ahora - ultima.cuando).total_seconds() < ANTI_REBOTE_MIN * 60:
                request.session["ultima_marca"] = {
                    "nombre": emp.nombre,
                    "documento": emp.documento,
                    "mensaje": f"Marcación reciente. Intenta nuevamente en {ANTI_REBOTE_MIN} min.",
                    "cuando": ultima.cuando.strftime("%Y-%m-%d %H:%M:%S"),
                }
                return redirect("success")

            # Intentar crear la marca del día (única por constraint)
            try:
                reg = Asistencia.objects.create(
                    empleado=emp,
                    fecha=hoy,
                    cuando=ahora,
                    dispositivo=request.META.get("REMOTE_ADDR", ""),
                )
                request.session["ultima_marca"] = {
                    "nombre": emp.nombre,
                    "documento": emp.documento,
                    "mensaje": "¡Entrada registrada!",
                    "cuando": reg.cuando.strftime("%Y-%m-%d %H:%M:%S"),
                }
            except IntegrityError:
                # Ya marcó hoy
                ya = emp.asistencias.filter(fecha=hoy).first()
                request.session["ultima_marca"] = {
                    "nombre": emp.nombre,
                    "documento": emp.documento,
                    "mensaje": "Ya existe una marcación de entrada para hoy.",
                    "cuando": ya.cuando.strftime("%Y-%m-%d %H:%M:%S") if ya else "",
                }
            return redirect("success")
    else:
        form = MarcarEntradaForm()

    return render(request, "login.html", {"form": form})

def success(request):
    data = request.session.pop("ultima_marca", None)
    return render(request, "success.html", {"data": data})
