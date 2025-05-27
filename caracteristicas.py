import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte

def extract_image_features(image_path, class_name, class_index):
    features = {
        "filename": os.path.basename(image_path),
        "class": class_name,
        "class_index": class_index
    }

    try:
        img = Image.open(image_path).convert("RGB")
        np_img = np.array(img)
        height, width, _ = np_img.shape
        features.update({
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 2),
            "is_square": width == height
        })

        # Color RGB
        for i, ch_name in enumerate(['R', 'G', 'B']):
            ch = np_img[:, :, i].flatten()
            features[f"mean_{ch_name}"] = np.mean(ch)
            features[f"std_{ch_name}"] = np.std(ch)
            features[f"min_{ch_name}"] = np.min(ch)
            features[f"max_{ch_name}"] = np.max(ch)
            features[f"skew_{ch_name}"] = skew(ch)
            features[f"kurt_{ch_name}"] = kurtosis(ch)
            hist, _ = np.histogram(ch, bins=16, range=(0, 255), density=True)
            for j, val in enumerate(hist):
                features[f"hist_{ch_name}_{j}"] = round(val, 5)

        # Color HSV
        hsv = rgb2hsv(np_img)
        for i, ch_name in enumerate(['H', 'S', 'V']):
            ch = hsv[:, :, i].flatten()
            features[f"mean_{ch_name}"] = np.mean(ch)
            features[f"std_{ch_name}"] = np.std(ch)
            features[f"skew_{ch_name}"] = skew(ch)
            features[f"kurt_{ch_name}"] = kurtosis(ch)

        # Escala de grises
        gray = rgb2gray(np_img)
        gray_u8 = img_as_ubyte(gray)
        features["entropy"] = shannon_entropy(gray)
        features["sobel"] = np.mean(sobel(gray))

        # GLCM
        glcm = graycomatrix(gray_u8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            vals = graycoprops(glcm, prop)[0]
            for i, angle in enumerate(["0", "45", "90", "135"]):
                features[f"{prop}_{angle}"] = round(vals[i], 5)

    except Exception as e:
        print(f"❌ Error en {image_path}: {e}")

    return features

# 📁 Ruta a tu dataset balanceado (ajústala)
dataset_path = r"C:\Users\bra\OneDrive - Universidad de Antioquia\Documentos\inteligencia\dataset_balanceado"

# Crear CSV con características
dataset = []
class_names = sorted(os.listdir(dataset_path))
class_map = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, fname)
            feats = extract_image_features(img_path, class_name, class_map[class_name])
            dataset.append(feats)

df = pd.DataFrame(dataset)
output_csv = os.path.join(dataset_path, "caracteristicas_300.csv")
df.to_csv(output_csv, index=False)

print(f"✅ CSV generado con {len(df)} imágenes y {df.shape[1]} características:")
print(output_csv)
