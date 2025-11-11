from django.urls import path
from . import views

urlpatterns = [
    path('', views.menu, name='menu'),
    path('registrar/', views.registrar, name='registrar'),
    path('ingresar/', views.ingresar, name='login'),
    path('success/', views.success, name='success'),
    path("camara/", views.camera_page, name="camera_page"),
    path("camara/stream/", views.video_feed, name="video_feed"),
    path("camera_status/", views.camera_status, name="camera_status"),
    path("finalizar_registro/", views.finalizar_registro, name="finalizar_registro"), 
]
