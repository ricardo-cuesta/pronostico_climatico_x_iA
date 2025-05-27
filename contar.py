import os

# Ruta al dataset balanceado
dataset_path = r"C:\Users\bra\OneDrive - Universidad de Antioquia\Documentos\inteligencia\dataset_balanceado"

print("📊 Distribución final de imágenes por clase:\n")

total_imagenes = 0

for class_name in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        num_images = len([
            fname for fname in os.listdir(class_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        total_imagenes += num_images
        print(f"🔹 {class_name}: {num_images} imágenes")

print(f"\n✅ Total de imágenes en el dataset: {total_imagenes}")
