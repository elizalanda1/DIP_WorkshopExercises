import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen desde el path proporcionado
image_path = "19_MR_s_output/19_MR_s1.jpg"
img_with_tumor = cv2.imread(image_path)

# Convertir la imagen de MRI con tumor a escala de grises
img_with_tumor_gray = cv2.cvtColor(img_with_tumor, cv2.COLOR_BGR2GRAY)

# Aplicar ecualización del histograma
img_eq = cv2.equalizeHist(img_with_tumor_gray)

# Función del algoritmo de crecimiento de regiones
def region_growing(img, seed, threshold=15):
    # Inicializar la máscara con ceros (mismo tamaño que la imagen)
    mask = np.zeros_like(img)
    
    # Obtener la intensidad del punto de semilla
    seed_intensity = img[seed]
    
    # Crear una lista para almacenar los píxeles que se van a examinar (comienza con el punto de semilla)
    pixel_list = [seed]
    
    # Direcciones para la conectividad de 4 vecinos
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Procesar la lista de píxeles (búsqueda en anchura)
    while pixel_list:
        # Sacar el último píxel
        x, y = pixel_list.pop()
        
        # Procesar el píxel solo si está dentro de la imagen y no ha sido añadido a la máscara
        if mask[x, y] == 0:
            # Marcar este píxel como parte de la región en la máscara
            mask[x, y] = 255
            
            # Comprobar los vecinos
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    # Si la intensidad del vecino está dentro del umbral
                    if abs(int(img[nx, ny]) - int(seed_intensity)) < threshold:
                        # Añadir el vecino a la lista de píxeles a procesar
                        pixel_list.append((nx, ny))
    
    return mask

# Función para manejar el clic del mouse
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Cuando se detecta un clic izquierdo, tomar esa posición como semilla
        seed_point = (y, x)  # cv2 usa coordenadas (y, x)
        print(f"Semilla seleccionada en: {seed_point}")
        
        # Aplicar el crecimiento de regiones desde el punto de semilla
        tumor_mask = region_growing(img_with_tumor_gray, seed_point)
        
        # Mostrar la máscara generada
        cv2.imshow("Mascara del Tumor", tumor_mask)

        # Crear la carpeta 'mask_output' en el mismo directorio de la imagen si no existe
        output_dir = os.path.join(os.path.dirname(image_path), 'mask_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Guardar la máscara con el nombre "mask_<x>_<y>.jpg" basado en la posición de la semilla
        mask_filename = f"mask_{seed_point[1]}_{seed_point[0]}.jpg"
        mask_path = os.path.join(output_dir, mask_filename)
        
        # Guardar la máscara como un archivo .jpg
        cv2.imwrite(mask_path, tumor_mask)
        print(f"Mascara guardada en: {mask_path}")

# Mostrar la imagen y esperar a que el usuario haga clic
cv2.imshow("Selecciona el punto de inicio", img_with_tumor_gray)
cv2.setMouseCallback("Selecciona el punto de inicio", mouse_click)

# Esperar a que el usuario cierre la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()