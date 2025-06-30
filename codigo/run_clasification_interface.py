import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

# Importar las clases desde el archivo search_classes.py
from dogdetection import DogDetectionClassificationPipeline
from gradio import DogDetectionPipelineInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Detección y Clasificación de Razas de Perros con Gradio."
    )
    parser.add_argument(
        "--detection_model_path",
        type=str,
        required=True,
        help="Ruta al modelo YOLO para detección (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--classification_model_path",
        type=str,
        required=True,
        help="Ruta al modelo Keras (e.g., .h5 o SavedModel) para clasificación de razas de perros.",
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        required=True,
        help="Ruta al archivo .npy con los nombres de todas las clases/razas (array NumPy de strings) que tu modelo de clasificación puede predecir.",
    )

    args = parser.parse_args()

    print("Cargando componentes...")

    # --- 1. Cargar el Modelo de Clasificación de Keras ---
    try:
        classification_model = load_model(args.classification_model_path)
        print(
            f"Modelo de clasificación cargado desde: {args.classification_model_path}"
        )
    except Exception as e:
        print(
            f"Error al cargar el modelo de clasificación Keras desde {args.classification_model_path}: {e}"
        )
        exit()

    # --- 2. Cargar Nombres de Clases ---
    try:
        class_names = np.load(
            args.class_names_path
        ).tolist()  # Convertir a lista de strings
        print(f"Nombres de clases cargados. {len(class_names)} clases.")
    except Exception as e:
        print(
            f"Error al cargar los nombres de las clases desde {args.class_names_path}: {e}"
        )
        exit()

    # --- 3. Inicializar el Pipeline de Detección y Clasificación ---
    # El modelo de detección YOLO se cargará internamente por la clase
    try:
        dog_pipeline = DogDetectionClassificationPipeline(
            detection_model_path=args.detection_model_path,
            classification_model=classification_model,
            class_names=class_names,
        )
        print("Pipeline de detección y clasificación inicializado.")
    except Exception as e:
        print(f"Error al inicializar el pipeline: {e}")
        print(
            "Asegúrate de que 'ultralytics' esté instalado y el modelo YOLO sea válido."
        )
        exit()

    # --- 4. Inicializar y Lanzar la Interfaz de Gradio ---
    print("Lanzando la interfaz de Gradio...")
    # Pasamos el pipeline al DogDetectionPipelineInterface para que lo use
    detection_interface = DogDetectionPipelineInterface(
        detection_model_path=args.detection_model_path,  # Se pasa de nuevo para la inicialización interna
        classification_model=classification_model,
        class_names=class_names,
    )
    detection_interface.launch(share=True)
    print("Interfaz de Gradio lanzada.")
