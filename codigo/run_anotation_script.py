import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
import cv2
import json  # Necesario para json.dump
from typing import Any

# Importar las clases desde el archivo search_classes.py
from dogdetection import DogDetectionClassificationPipeline


class AnnotationExporter:
    @staticmethod
    def to_yolo(
        image_width: int,
        image_height: int,
        predictions: list[tuple[str, tuple[int, int, int, int]]],
        class_map: dict[str, int],
    ) -> list[str]:
        """Convierte las predicciones a formato YOLO."""
        yolo_annotations = []

        for class_name, (x1, y1, x2, y2) in predictions:
            # Convertir a coordenadas normalizadas YOLO
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            class_id = class_map[class_name]
            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        return yolo_annotations

    @staticmethod
    def to_coco(
        images: list[dict], annotations: list[dict], categories: list[dict]
    ) -> dict[str, Any]:
        """Crea un archivo COCO JSON con las anotaciones."""
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {},
            "licenses": [],
        }


def script_anotacion(
    pipeline: DogDetectionClassificationPipeline,
    input_folder: str,
    output_folder: str,
    class_map: dict[str, int],
    save_annotated_images: bool = True,
) -> None:
    """Procesa todas las imágenes en una carpeta y exporta anotaciones.

    Args:
        pipeline: Pipeline de detección/clasificación
        input_folder (str): Carpeta con imágenes de entrada
        output_folder (str): Carpeta para guardar resultados
        class_map (Dict): Mapeo de nombre de clase a ID
        save_annotated_images (bool): Si guardar imágenes anotadas
    """
    print(f"Iniciando procesamiento por lotes de imágenes en: {input_folder}")

    # Crear carpetas de salida si no existen
    os.makedirs(output_folder, exist_ok=True)
    yolo_folder = os.path.join(output_folder, "yolo_annotations")
    coco_folder = os.path.join(output_folder, "coco_annotations")
    os.makedirs(yolo_folder, exist_ok=True)
    os.makedirs(coco_folder, exist_ok=True)

    if save_annotated_images:
        annotated_folder = os.path.join(output_folder, "annotated_images")
        os.makedirs(annotated_folder, exist_ok=True)

    # Preparar estructura COCO
    coco_images = []
    coco_annotations = []
    coco_categories = [{"id": v, "name": k} for k, v in class_map.items()]
    annotation_id_counter = 1  # Usar un contador único para IDs de anotaciones

    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print(f"No se encontraron imágenes en la carpeta de entrada: {input_folder}")
        return

    # Procesar cada imagen en la carpeta de entrada
    for i, filename in enumerate(image_files):
        image_path = os.path.join(input_folder, filename)
        print(f"Procesando imagen {i + 1}/{len(image_files)}: {filename}")

        # cv2.imread lee en BGR. YOLO y Gradio suelen esperar RGB.
        # Asegúrate de la consistencia. Si tu pipeline YOLO/Keras fue entrenado con RGB, convierte.
        image = cv2.imread(image_path)
        if image is None:
            print(f"ADVERTENCIA: No se pudo leer la imagen {filename}. Saltando.")
            continue

        # Convertir a RGB si tu pipeline espera RGB (Gradio suele usar RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_height, image_width = image_rgb.shape[
            :2
        ]  # Usar la imagen RGB para obtener dimensiones

        # Procesar imagen con el pipeline
        # `predictions` ahora es `list[dict]` del formato devuelto por DogDetectionClassificationPipeline
        annotated_image_rgb, predictions = pipeline.process_image(image_rgb)

        # Guardar imagen anotada si se requiere (convertir de RGB a BGR para cv2.imwrite)
        if save_annotated_images:
            output_path = os.path.join(annotated_folder, filename)
            cv2.imwrite(
                output_path, cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
            )

        # Exportar a YOLO format
        yolo_annos = AnnotationExporter.to_yolo(
            image_width, image_height, predictions, class_map
        )
        yolo_path = os.path.join(yolo_folder, os.path.splitext(filename)[0] + ".txt")
        with open(yolo_path, "w") as f:
            f.write("\n".join(yolo_annos))

        # Preparar datos para COCO
        image_id = i + 1  # ID de imagen basado en el índice
        coco_images.append(
            {
                "id": image_id,
                "width": image_width,
                "height": image_height,
                "file_name": filename,
                "license": 0,  # Placeholder
                "date_captured": "",  # Placeholder
            }
        )

        for pred in predictions:  # predictions ya es list[dict]
            class_name = pred["class"]
            x1, y1, x2, y2 = pred["bbox"]
            width = x2 - x1
            height = y2 - y1

            coco_annotations.append(
                {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": class_map[class_name],
                    "bbox": [x1, y1, width, height],
                    "area": float(width * height),  # COCO espera float para 'area'
                    "segmentation": [],  # No se genera segmentación
                    "iscrowd": 0,
                }
            )
            annotation_id_counter += 1

    # Guardar COCO JSON
    coco_data = AnnotationExporter.to_coco(
        coco_images, coco_annotations, coco_categories
    )
    coco_path = os.path.join(coco_folder, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nProcesamiento completado. Resultados guardados en: {output_folder}")
    print(f"  - Anotaciones YOLO en: {yolo_folder}")
    print(f"  - Anotaciones COCO en: {coco_folder}")
    if save_annotated_images:
        print(f"  - Imágenes anotadas en: {annotated_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Herramienta de Anotación por Lotes de Razas de Perros."
    )
    parser.add_argument(
        "--detection_model_path",
        type=str,
        required=True,
        help="Ruta al modelo YOLO para detección (e.g., yolov8n.pt).",
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
        help="Ruta al archivo .npy con los nombres de todas las clases/razas (array NumPy de strings).",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Ruta a la carpeta que contiene las imágenes de entrada para anotar.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Ruta a la carpeta donde se guardarán las anotaciones y las imágenes procesadas.",
    )
    parser.add_argument(
        "--save_annotated_images",
        action="store_true",
        help="Bandera para guardar las imágenes con las anotaciones dibujadas.",
    )

    args = parser.parse_args()

    print("--- Inicializando Herramienta de Anotación por Lotes ---")

    try:
        classification_model = load_model(args.classification_model_path)
        print(
            f"Modelo de clasificación cargado desde: {args.classification_model_path}"
        )

        class_names = np.load(args.class_names_path).tolist()
        print(f"Nombres de clases cargados. {len(class_names)} clases.")

        # Crear un mapeo de nombre de clase a ID (necesario para YOLO y COCO)
        class_map = {name: idx for idx, name in enumerate(class_names)}
        print("Mapeo de clases creado.")

        pipeline = DogDetectionClassificationPipeline(
            detection_model_path=args.detection_model_path,
            classification_model=classification_model,
            class_names=class_names,  # Las clases se usan en DogDetectionClassificationPipeline para obtener el nombre
        )
        print("Pipeline de detección y clasificación inicializado.")

        script_anotacion(
            pipeline=pipeline,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            class_map=class_map,
            save_annotated_images=args.save_annotated_images,
        )
        print("Proceso de anotación por lotes finalizado.")

    except Exception as e:
        print(f"ERROR: No se pudo ejecutar la herramienta de anotación. Detalles: {e}")
        print(
            "Asegúrate de que 'ultralytics' esté instalado y los paths de los modelos y carpetas sean correctos."
        )
        exit()
