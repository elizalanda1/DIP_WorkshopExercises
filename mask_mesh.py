import cv2
import os
import numpy as np

# Ruta a la carpeta con las máscaras
mask_folder = "19_MR_s_output/mask_output"

# Listar todas las imágenes en la carpeta
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Verificar que haya imágenes en la carpeta
if len(mask_files) == 0:
    print("No se encontraron imágenes en la carpeta.")
else:
    # Inicializar la variable que almacenará la suma de las imágenes
    summed_mask = None
    
    # Iterar sobre todas las imágenes en la carpeta
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Leer la máscara en escala de grises
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if summed_mask is None:
            # Si es la primera imagen, inicializar summed_mask con el tamaño adecuado
            summed_mask = np.zeros_like(mask, dtype=np.float32)
        
        # Sumar la máscara actual a la imagen acumulada
        summed_mask += mask
    
    # Normalizar la imagen sumada para que los valores estén entre 0 y 255
    summed_mask = np.clip(summed_mask, 0, 255).astype(np.uint8)

    # Mostrar la imagen final
    cv2.imshow("Imagen Final Suma de Mascaras", summed_mask)
    
    # Guardar la imagen final en la carpeta
    output_path = os.path.join(mask_folder, "mask_sum.jpg")
    cv2.imwrite(output_path, summed_mask)
    print(f"Imagen final guardada en: {output_path}")
    
    # Esperar a que el usuario cierre la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()
