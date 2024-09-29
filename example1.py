import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = "example1.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: No se pudo cargar la imagen desde {image_path}")
else:
    # Convertir la imagen de BGR a escala de grises (BGR -> GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convertir la imagen de BGR a RGB (BGR -> RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mostrar las imágenes originales y convertidas usando matplotlib
    fig, ax = plt.subplots(3, 1, figsize=(7, 7))

    # Imagen original en BGR
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original (BGR)")
    ax[0].axis('off')

    # Imagen en escala de grises
    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title("Escala de Grises (GRAY)")
    ax[1].axis('off')

    # Imagen en RGB
    ax[2].imshow(rgb_image)
    ax[2].set_title("RGB")
    ax[2].axis('off')

    # Mostrar las imágenes
    plt.show()
