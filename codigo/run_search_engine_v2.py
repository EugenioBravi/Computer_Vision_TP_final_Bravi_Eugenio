import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os

# Importar las clases desde el archivo search_classes.py
from dogsearch import DogSearchEngineV2
from gradio import DogSearchInterfaceV2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buscador Multi-Modelo de Razas de Perros con Gradio."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Ruta al directorio que contiene los archivos de modelos Keras (.h5 o SavedModel).",
    )
    parser.add_argument(
        "--indexes_dir",
        type=str,
        required=True,
        help="Ruta al directorio que contiene los archivos de índice FAISS (.faiss).",
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
    parser.add_argument(
        "--class_names_path",
        type=str,
        required=True,
        help="Ruta al archivo .npy con los nombres de todas las clases/razas (array NumPy de strings).",
    )

    args = parser.parse_args()

    print("Cargando componentes...")

    # --- 1. Cargar Imágenes, Etiquetas y Nombres de Clases ---
    try:
        all_images = np.load(args.images_path)
        all_labels = np.load(args.labels_path)
        class_names = np.load(
            args.class_names_path
        ).tolist()  # Convertir a lista si es necesario
        print(f"Imágenes y etiquetas cargadas. {len(all_images)} registros.")
        print(f"Clases disponibles: {len(class_names)}")
    except Exception as e:
        print(f"Error al cargar imágenes, etiquetas o nombres de clases: {e}")
        exit()

    # --- 2. Cargar Múltiples Modelos e Índices ---
    models_indexs_dict = {}

    # Obtener la lista de modelos disponibles en el directorio
    model_files = [
        f
        for f in os.listdir(args.models_dir)
        if f.endswith(".h5")
        or os.path.isdir(os.path.join(args.models_dir, f))
        and not f.startswith(".")
    ]

    if not model_files:
        print(f"Error: No se encontraron modelos en el directorio {args.models_dir}.")
        exit()

    for model_file in model_files:
        model_name_base = (
            os.path.splitext(model_file)[0]
            if model_file.endswith(".h5")
            else model_file
        )  # Nombre sin extensión o nombre de directorio para SavedModel
        model_path = os.path.join(args.models_dir, model_file)
        index_path = os.path.join(
            args.indexes_dir, model_name_base + ".bin"
        )  # Asume el mismo nombre + .bin para el índice

        try:
            print(f"Cargando modelo '{model_name_base}' desde {model_path}...")
            model = load_model(model_path)
            # Asegúrate de que el modelo extraiga características. Ajusta la dimensión de salida si es necesario.
            feature_dim = model.output_shape[-1]
            print(
                f"Modelo '{model_name_base}' cargado. Dimensión de características: {feature_dim}"
            )

            print(f"Cargando índice para '{model_name_base}' desde {index_path}...")
            index = faiss.read_index(index_path)
            print(
                f"Índice para '{model_name_base}' cargado. Contiene {index.ntotal} vectores."
            )
            if index.d != feature_dim:
                print(
                    f"ADVERTENCIA: La dimensión del índice ({index.d}) para '{model_name_base}' no coincide con la del modelo ({feature_dim})."
                )

            models_indexs_dict[model_name_base] = (model, index)

        except Exception as e:
            print(f"Error al cargar modelo o índice para '{model_name_base}': {e}")
            # Puedes optar por continuar con otros modelos o salir
            continue  # Intenta cargar el siguiente par de modelo/índice

    if not models_indexs_dict:
        print("ERROR FATAL: No se pudo cargar ningún modelo/índice válido. Saliendo.")
        exit()

    # --- 3. Inicializar el Motor de Búsqueda V2 ---
    search_engine_v2 = DogSearchEngineV2(
        model_index=models_indexs_dict,
        images=all_images,
        labels=all_labels,
        class_names=class_names,  # Pasa los nombres de las clases
    )
    print("Motor de búsqueda V2 inicializado.")

    # --- 4. Inicializar y Lanzar la Interfaz de Gradio V2 ---
    print("Lanzando la interfaz de Gradio V2...")
    dog_interface_v2 = DogSearchInterfaceV2(search_engine=search_engine_v2)
    dog_interface_v2.launch(
        share=True
    )  # `share=True` para un enlace público temporal, `debug=True` para ver logs.
    print("Interfaz de Gradio V2 lanzada.")
