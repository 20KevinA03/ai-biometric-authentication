from django.urls import path
from . import views

urlpatterns = [
    path("", views.menu, name="menu"),
    path("registrar/", views.registrar, name="registrar"),
    path("finalizar-registro/", views.finalizar_registro, name="finalizar_registro"),
    path("ingresar/", views.ingresar, name="ingresar"),
    path("finalizar-ingreso/", views.finalizar_ingreso, name="finalizar_ingreso"),
    path("video-feed/", views.video_feed, name="video_feed"),
    path("camera-status/", views.camera_status, name="camera_status"),
    path("stop-camera/", views.stop_camera, name="stop_camera"),
    path("face/<str:documento>.png", views.face_image, name="face_image"),
    path("success/", views.success, name="success"),
]
