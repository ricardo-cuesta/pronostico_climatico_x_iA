import os
from collections import Counter
from PIL import Image

dataset_path = r"C:\Users\bra\OneDrive - Universidad de Antioquia\Documentos\inteligencia\archive\Multi-class Weather Dataset"
classes = os.listdir(dataset_path)
conteo = {clase: len(os.listdir(os.path.join(dataset_path, clase))) for clase in classes}

print("Clases disponibles:", classes)
print("Cantidad de imágenes por clase:", conteo)

# Ver una imagen de muestra
sample_path = os.path.join(dataset_path, classes[0], os.listdir(os.path.join(dataset_path, classes[0]))[0])
img = Image.open(sample_path)
print("Tamaño de imagen:", img.size)
