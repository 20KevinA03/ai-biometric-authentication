from django import forms
from .models import Empleado


class RegistroEmpleadoForm(forms.ModelForm):
    class Meta:
        model = Empleado
        fields = ["nombre", "documento"]
        widgets = {
            "nombre": forms.TextInput(attrs={
                "placeholder": "Nombre",
                "autocomplete": "off",
            }),
            "documento": forms.TextInput(attrs={
                "placeholder": "Documento",
                "autocomplete": "off",
                "inputmode": "numeric",
            }),
        }

    def clean_documento(self):
        doc = self.cleaned_data["documento"].strip()
        if Empleado.objects.filter(documento=doc).exists():
            raise forms.ValidationError("Este documento ya está registrado.")
        return doc


class MarcarEntradaForm(forms.Form):
    documento = forms.CharField(
        label="",
        widget=forms.TextInput(attrs={
            "placeholder": "Documento",
            "autocomplete": "off",
        })
    )

    nombre = forms.CharField(
        label="",
        required=False,
        widget=forms.TextInput(attrs={
            "placeholder": "Nombre (opcional)",
            "autocomplete": "off",
        })
    )
