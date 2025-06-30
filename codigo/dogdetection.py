from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
import time


class DogDetectionClassificationPipeline:
    def __init__(self, detection_model: str, model: Model, class_names):
        # Cargar modelo YOLO para detección
        self.detector = YOLO(detection_model)
        self.model = model
        self._class_names = sorted(class_names)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa una imagen para el modelo"""
        image = cv2.resize(image, (144, 144))
        image = preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(
        self, input_image: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int, int]]]]:
        """Procesa una imagen completa detectando y clasificando perros"""

        # Detección de perros con YOLO
        results = self.detector(input_image)

        # Lista para almacenar resultados
        predictions = []
        annotated_image = input_image.copy()

        # Procesar cada detección
        for result in results:
            for box in result.boxes:
                # Verificar que sea un perro (clase 16 en YOLO)
                if int(box.cls) == 16:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Recortar el perro de la imagen
                    dog_crop = input_image[y1:y2, x1:x2]
                    dog_crop = self._preprocess_image(dog_crop)
                    # Clasificar la raza
                    predicted_class = self.model.predict(dog_crop)
                    predicted_class = self._class_names[np.argmax(predicted_class)]

                    # Dibujar bounding box y etiqueta
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_image,
                        predicted_class,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    predictions.append((predicted_class, (x1, y1, x2, y2)))

        return annotated_image, predictions


class classify_image_tflite:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def predict(self, image: np.ndarray):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        start_time = time.perf_counter()

        self.interpreter.set_tensor(input_details[0]["index"], image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]["index"])

        end_time = time.perf_counter()

        # Cálculo del tiempo transcurrido en milisegundos
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Tiempo de inferencia: {elapsed_time_ms:.2f} milisegundos")

        return output
