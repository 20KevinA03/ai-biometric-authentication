from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User

class NombreDocumentoBackend(BaseBackend):
    # authenticate se llama con los kwargs que tú pases desde la vista
    def authenticate(self, request, nombre=None, documento=None, **kwargs):
        if not nombre or not documento:
            return None
        try:
            user = User.objects.get(username=nombre)
        except User.DoesNotExist:
            return None
        # valida contra el documento guardado en Profile
        if hasattr(user, 'profile') and user.profile.documento == documento:
            return user
        return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
