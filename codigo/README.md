# 🐶 Aplicaciones Gradio para Perros

Este repositorio contiene tres aplicaciones de Gradio para trabajar con imágenes de perros:

1.  **Buscador de Razas de Perros:** Encuentra perros similares en una base de datos usando un solo modelo.
2.  **Buscador Multi-Modelo de Razas de Perros:** Similar al anterior, pero permite seleccionar entre múltiples modelos para la búsqueda.
3.  **Detección y Clasificación de Razas de Perros:** Detecta perros en imágenes y clasifica su raza.

# Requisitos Previos

- Python 3.8 o superior

## Instalación

1.  **Clona el repositorio o crea los archivos:**
    Asegúrate de tener todos los archivos (`.py` y `README.md`) en el mismo directorio.

2.  **Crea un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    # En Windows: .\venv\Scripts\activate
    # En macOS/Linux: source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    instala:
    ```bash
    pip install -r requirements.txt
    ```

## Preparación de Datos (¡Crucial!)

Todas las aplicaciones requieren archivos de datos específicos (modelos, índices, imágenes, etiquetas). Debes preparar estos archivos antes de ejecutar los scripts.

### Archivos Comunes Requeridos:

- **Imágenes de Base de Datos (`all_dog_images.npy`):** Archivo NumPy `(N, H, W, 3)` de tipo `uint8` (0-255).
- **Etiquetas de Imágenes (`all_dog_labels.npy`):** Archivo NumPy `(N,)` de `str`.
- **Nombres de Clases/Razas (`all_class_names.npy`):** Archivo NumPy `(Número_de_clases,)` de `str`.

### Archivos Específicos por Aplicación:

#### Para `run_search_engine.py` (Buscador Simple):

- **Modelo de Extracción de Características (`my_feature_extractor.h5`):** Modelo Keras (`.h5` o SavedModel).
- **Índice FAISS (`my_faiss_index.bin`):** Archivo FAISS binario.

#### Para `run_search_engine_v2.py` (Buscador Multi-Modelo):

- **Directorio de Modelos (`/ruta/a/modelos_busqueda/`):** Contiene múltiples modelos Keras (ej. `modelo1.h5`, `modelo2.h5`).
- **Directorio de Índices FAISS (`/ruta/a/indices_busqueda/`):** Contiene índices FAISS correspondientes (ej. `modelo1.bin`, `modelo2.bin`). **Los nombres deben coincidir con los de los modelos.**

#### Para `run_classification_interface.py` (Detección/Clasificación):

- **Modelo de Detección YOLO (`yolov8n.pt`):** Archivo de pesos YOLO (`.pt`). Descargar de la web de Ultralytics.
- **Modelo de Clasificación de Razas (`my_dog_breed_classifier.h5`):** Modelo Keras (`.h5` o SavedModel) para clasificar razas individuales.

## Ejecución de las Aplicaciones

Cada script se ejecuta de forma independiente y requiere rutas a sus archivos de datos.

### 1. Buscador de Razas de Perros (Simple)

```bash
python run_search_engine.py \
    --model_path /ruta/a/tu/modelo/my_feature_extractor.h5 \
    --index_path /ruta/a/tu/indice/my_faiss_index.bin \
    --images_path /ruta/a/tus/imagenes/all_dog_images.npy \
    --labels_path /ruta/a/tus/etiquetas/all_dog_labels.npy
```

### 2. Buscador Multi-Modelo de Razas de Perros

```bash
python run_search_engine_v2.py \
    --models_dir /ruta/a/tus/directorio_modelos_busqueda \
    --indexes_dir /ruta/a/tus/directorio_indices_busqueda \
    --images_path /ruta/a/tus/imagenes/all_dog_images.npy \
    --labels_path /ruta/a/tus/etiquetas/all_dog_labels.npy \
    --class_names_path /ruta/a/tus/clases/all_class_names.npy
```

### 3. Detección y Clasificación de Razas de Perros

```bash
python run_classification_interface.py \
    --detection_model_path /ruta/a/tu/yolo/yolov8n.pt \
    --classification_model_path /ruta/a/tu/modelo/my_dog_breed_classifier.h5 \
    --class_names_path /ruta/a/tus/clases/all_breed_names.npy
```

### 4. Detección y Anotación por Lotes

```bash
python run_anotation_script.py \
    --detection_model_path /ruta/a/tu/yolo/yolov8n.pt \
    --classification_model_path /ruta/a/tu/modelo/my_dog_breed_classifier.h5 \
    --class_names_path /ruta/a/tus/clases/all_class_names.npy \
    --input_folder /ruta/a/tu/carpeta_imagenes_entrada \
    --output_folder /ruta/a/tu/carpeta_resultados_anotacion \
    --save_annotated_images # Opcional: para guardar también las imágenes con anotaciones dibujadas
```
