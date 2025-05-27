import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# === CONFIGURACIÓN ===

input_dir = r"C:\Users\bra\OneDrive - Universidad de Antioquia\Documentos\inteligencia\archive\Multi-class Weather Dataset"
output_dir = r"C:\Users\bra\OneDrive - Universidad de Antioquia\Documentos\inteligencia\dataset_balanceado"
os.makedirs(output_dir, exist_ok=True)

# Meta por clase (mínimo y máximo)
target_min = 350
target_max = 400

# === GENERADOR DE AUMENTO ===

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === COPIA TODAS LAS IMÁGENES ORIGINALES ===

for class_name in os.listdir(input_dir):
    src_class_path = os.path.join(input_dir, class_name)
    dst_class_path = os.path.join(output_dir, class_name)
    os.makedirs(dst_class_path, exist_ok=True)

    for fname in os.listdir(src_class_path):
        shutil.copy2(os.path.join(src_class_path, fname), dst_class_path)

# === AUMENTAR CLASES NECESARIAS (USANDO IMÁGENES ALEATORIAS) ===

for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    original_images = os.listdir(class_input_path)
    original_count = len(original_images)

    if original_count >= target_min:
        print(f"✅ {class_name} ya tiene {original_count} imágenes (OK)")
        continue

    to_generate = target_min - original_count
    print(f"➕ Generando {to_generate} imágenes para clase '{class_name}'...")

    generated = 0
    while generated < to_generate:
        random.shuffle(original_images)  # Mezcla aleatoriamente la lista de imágenes

        for img_name in original_images:
            img_path = os.path.join(class_input_path, img_name)

            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=class_output_path,
                                          save_prefix='aug',
                                          save_format='jpg'):
                    generated += 1
                    break  # solo una imagen por batch

                if generated >= to_generate:
                    break

            except Exception as e:
                print(f"Error con {img_path}: {e}")

print("🎯 Aumento completado con imágenes aleatorias.")
