from django.db import models
from django.utils import timezone

class Empleado(models.Model):
    nombre = models.CharField(max_length=150)
    documento = models.CharField(max_length=30, unique=True)
    activo = models.BooleanField(default=True)
    creado = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["nombre"]

    def __str__(self):
        return f"{self.nombre} ({self.documento})"


class Asistencia(models.Model):
    empleado = models.ForeignKey(Empleado, on_delete=models.CASCADE, related_name="asistencias")
    # fecha del día laboral (garantiza 1 marca por día)
    fecha = models.DateField(default=timezone.localdate, db_index=True)
    # timestamp exacto de la marcación
    cuando = models.DateTimeField(default=timezone.now)
    dispositivo = models.CharField(max_length=50, blank=True) 

    class Meta:
        ordering = ["-cuando"]
        constraints = [
            models.UniqueConstraint(fields=["empleado", "fecha"], name="unica_marca_por_dia")
        ]

    def __str__(self):
        return f"{self.empleado} - {self.fecha} {self.cuando:%H:%M:%S}"
