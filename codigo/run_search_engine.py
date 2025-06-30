import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

# Importar las clases desde el archivo search_classes.py
from dogsearch import DogSearchEngine
from gradio import DogSearchInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buscador de Razas de Perros con Gradio."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo Keras (e.g., .h5 o SavedModel) para extracción de características.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Ruta al archivo de índice FAISS (.faiss).",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Ruta al archivo .npy con todas las imágenes pre-cargadas (array NumPy).",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Ruta al archivo .npy con todas las etiquetas de las imágenes (array NumPy de strings).",
    )

    args = parser.parse_args()

    print("Cargando componentes...")

    # --- 1. Cargar el Modelo Keras ---
    try:
        feature_extractor_model = load_model(args.model_path)
        # Asegúrate de que la dimensión de salida de tu modelo sea la correcta para FAISS
        # Si tu modelo tiene una capa de pooling final, feature_dim será la dimensión del vector.
        feature_dim = feature_extractor_model.output_shape[-1]
        print(
            f"Modelo cargado desde: {args.model_path}. Dimensión de características: {feature_dim}"
        )
    except Exception as e:
        print(f"Error al cargar el modelo Keras desde {args.model_path}: {e}")
        exit()

    # --- 2. Cargar el Índice FAISS ---
    try:
        faiss_index = faiss.read_index(args.index_path)
        print(
            f"Índice FAISS cargado desde: {args.index_path}. Contiene {faiss_index.ntotal} vectores."
        )
        if faiss_index.d != feature_dim:
            print(
                f"ADVERTENCIA: La dimensión del índice FAISS ({faiss_index.d}) no coincide con la del modelo ({feature_dim}). Esto podría causar errores."
            )
    except Exception as e:
        print(f"Error al cargar el índice FAISS desde {args.index_path}: {e}")
        exit()

    # --- 3. Cargar Imágenes y Etiquetas ---
    try:
        all_images = np.load(args.images_path)
        all_labels = np.load(args.labels_path)
        print(f"Imágenes y etiquetas cargadas. {len(all_images)} registros.")
        if (
            len(all_images) != faiss_index.ntotal
            or len(all_labels) != faiss_index.ntotal
        ):
            print(
                f"ADVERTENCIA: El número de imágenes/etiquetas no coincide con el número de vectores en el índice. {len(all_images)} imágenes, {len(all_labels)} etiquetas, {faiss_index.ntotal} vectores en índice."
            )
    except Exception as e:
        print(f"Error al cargar imágenes o etiquetas: {e}")
        exit()

    # --- 4. Inicializar el Motor de Búsqueda ---
    search_engine = DogSearchEngine(
        model=feature_extractor_model,
        index=faiss_index,
        images=all_images,
        labels=all_labels,
    )
    print("Motor de búsqueda inicializado.")

    # --- 5. Inicializar y Lanzar la Interfaz de Gradio ---
    print("Lanzando la interfaz de Gradio...")
    dog_interface = DogSearchInterface(search_engine=search_engine)
    dog_interface.launch(
        share=True
    )  # `share=True` para un enlace público temporal, `debug=True` para ver logs.
    print("Interfaz de Gradio lanzada.")
