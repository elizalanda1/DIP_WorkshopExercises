import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para aplicar las operaciones morfológicas
def apply_morphological_operations(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# Cargar las imágenes usando el path correcto
img1_gray = apply_morphological_operations("example3/1.jpg")  # Para dilatación y erosión
img2_gray = apply_morphological_operations("example3/2.jpg")  # Para opening
img3_gray = apply_morphological_operations("example3/3.jpg")  # Para closing

# Generar el kernel de 5x5
kernel = np.ones((5, 5), np.uint8)

# Aplicar las operaciones morfológicas
img_dilated = cv2.dilate(img1_gray, kernel, iterations=1)
img_eroded = cv2.erode(img1_gray, kernel, iterations=1)
img_opening = cv2.morphologyEx(img2_gray, cv2.MORPH_OPEN, kernel)
img_closing = cv2.morphologyEx(img3_gray, cv2.MORPH_CLOSE, kernel)

# Mostrar la imagen original 1 y sus resultados
plt.figure(figsize=(12, 6))

# Mostrar la imagen original 1
plt.subplot(1, 3, 1)
plt.imshow(img1_gray, cmap='gray')
plt.title("Imagen Original 1")
plt.axis('off')

# Mostrar la imagen dilatada
plt.subplot(1, 3, 2)
plt.imshow(img_dilated, cmap='gray')
plt.title("Imagen Dilatada")
plt.axis('off')

# Mostrar la imagen erosionada
plt.subplot(1, 3, 3)
plt.imshow(img_eroded, cmap='gray')
plt.title("Imagen Erosionada")
plt.axis('off')

plt.tight_layout()
plt.show()

# Mostrar la imagen original 2 y su resultado de opening
plt.figure(figsize=(9, 6))

# Mostrar la imagen original 2
plt.subplot(1, 2, 1)
plt.imshow(img2_gray, cmap='gray')
plt.title("Imagen Original 2")
plt.axis('off')

# Mostrar la imagen con opening
plt.subplot(1, 2, 2)
plt.imshow(img_opening, cmap='gray')
plt.title("Imagen con Opening")
plt.axis('off')

plt.tight_layout()
plt.show()

# Mostrar la imagen original 3 y su resultado de closing
plt.figure(figsize=(9, 6))

# Mostrar la imagen original 3
plt.subplot(1, 2, 1)
plt.imshow(img3_gray, cmap='gray')
plt.title("Imagen Original 3")
plt.axis('off')

# Mostrar la imagen con closing
plt.subplot(1, 2, 2)
plt.imshow(img_closing, cmap='gray')
plt.title("Imagen con Closing")
plt.axis('off')

plt.tight_layout()
plt.show()
