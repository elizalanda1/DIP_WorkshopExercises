import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Función para mostrar imagen
def show_image(img, title, position):
    plt.subplot(position)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Función para mostrar histograma con título específico
def show_histogram(img, title, position):
    plt.subplot(position)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title(f"Histograma de {title}")
    plt.xlim([0, 256])
    plt.ylim([0, 15000])  # Establecer el límite en el eje y

# Cargar la imagen en escala de grises
image_path = "example2.jpg"
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Error: No se pudo cargar la imagen desde {image_path}")
else:
    # Saturar valores de gris por encima de 150
    img_high_saturation = img_gray.copy()
    img_high_saturation[img_high_saturation > 150] = 255

    # Saturar valores de gris por debajo de 100
    img_low_saturation = img_gray.copy()
    img_low_saturation[img_low_saturation < 100] = 0

    # Aplicar ecualización del histograma a la imagen original
    img_eq = cv2.equalizeHist(img_gray)

    # Crear la figura para las imágenes
    plt.figure(figsize=(10, 7))

    # Crear un objeto gridspec para las imágenes
    gs_images = gridspec.GridSpec(2, 2)

    # Mostrar las imágenes
    show_image(img_gray, "Imagen Original", gs_images[0, 0])
    show_image(img_high_saturation, "Saturada (Valores > 150)", gs_images[0, 1])
    show_image(img_low_saturation, "Saturada (Valores < 100)", gs_images[1, 0])
    show_image(img_eq, "Imagen Ecualizada", gs_images[1, 1])

    # Ajustar el layout de las imágenes
    plt.tight_layout()
    plt.show(block=False)  # Mostrar la figura de imágenes sin bloquear

    # Crear la figura para los histogramas
    plt.figure(figsize=(10, 7))

    # Crear un objeto gridspec para los histogramas
    gs_histograms = gridspec.GridSpec(2, 2)

    # Mostrar los histogramas
    show_histogram(img_gray, "Imagen Original", gs_histograms[0, 0])
    show_histogram(img_high_saturation, "Saturada (Valores > 150)", gs_histograms[0, 1])
    show_histogram(img_low_saturation, "Saturada (Valores < 100)", gs_histograms[1, 0])
    show_histogram(img_eq, "Imagen Ecualizada", gs_histograms[1, 1])

    # Ajustar el layout de los histogramas
    plt.tight_layout()
    plt.show()  # Mostrar la figura de histogramas
