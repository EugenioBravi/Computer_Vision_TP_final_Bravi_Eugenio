import numpy as np
import tensorflow as tf
import cv2
import faiss
from sklearn.metrics import ndcg_score
from tensorflow.keras import Model
from faiss import Index
from tensorflow.keras.applications.resnet50 import preprocess_input


class IndexBuilder:
    def __init__(self, dataset: np.ndarray):
        self._dataset: np.ndarray = np.copy(dataset)

    # 1. Extraer embeddings de todas las imágenes
    def _extract_features(self, model: Model, batch_size: int = 64) -> np.ndarray:
        print("Extrayendo características")
        num_images = len(self._dataset)
        all_features = []
        for i in range(0, num_images, batch_size):
            batch = self._dataset[i : i + batch_size]
            # Predicción del batch
            batch_features = model.predict(batch, verbose=0)
            all_features.append(batch_features)

        # Concatenar todos los batches
        return np.concatenate(all_features, axis=0)

    # 2. Construir índice FAISS
    def build_faiss_index(self, model: Model) -> Index:
        features = self._extract_features(model)
        print("Construyendo índice FAISS...")
        dimension = features.shape[
            1
        ]  # Dimensión de los embeddings (2048 para ResNet50)
        index = faiss.IndexFlatL2(dimension)  # Usamos distancia L2 (Euclidiana)
        index.add(features.astype("float32"))  # FAISS requiere float32
        return index


class DogSearchEngine:
    def __init__(
        self,
        model: Model,
        index: Index,
        images: np.ndarray[np.ndarray],
        labels: np.ndarray[str],
    ):
        self.model = model
        self.index = index
        self.all_images = images
        self.all_labels = labels
        self.image_size = (144, 144)

    def search(
        self, input_image: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, list[np.ndarray], list[str], list[float]]:
        """Busca perros similares en la base de datos"""
        # Preprocesar imagen de entrada
        processed_img = self._preprocess_image(input_image)
        input_tensor = tf.expand_dims(processed_img, axis=0)
        # input_tensor = preprocess_input(input_tensor)
        # Extraer características
        input_features = self.model.predict(input_tensor, verbose=0).astype("float32")

        # Buscar imágenes similares
        distances, indices = self.index.search(input_features, k)

        # Obtener resultados
        similar_images = self.all_images[indices[0]]
        similar_classes = [self.all_labels[idx] for idx in indices[0]]
        distances = distances[0]

        # Convertir imágenes a formato adecuado para Gradio (uint8 [0-255])
        similar_images = [(img).astype(np.uint8) for img in similar_images]

        return input_image, similar_images, similar_classes, distances

    def evaluate(
        self, image: np.ndarray, k=10
    ) -> tuple[np.ndarray, list[np.ndarray], list[str], list[float], str]:
        """Evalúa la imagen y determina la raza más probable por voto mayoritario"""
        # Realizar la búsqueda de imágenes similares
        input_image, similar_images, similar_classes, distances = self.search(image, k)

        # Contar las ocurrencias de cada clase
        class_counts = {}
        for class_name in similar_classes:
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        # Obtener la clase con mayor conteo
        predicted_class = max(class_counts.items(), key=lambda x: x[1])[0]

        return input_image, similar_images, similar_classes, distances, predicted_class

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa una imagen para el modelo"""
        image = cv2.resize(image, self.image_size)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return preprocess_input(image)

    def ndcg(
        self, images: np.ndarray[np.ndarray], true_labels: np.ndarray[str], k: int = 10
    ):
        ndcg_scores = []

        for img, true_label in zip(images, true_labels):
            # Obtener resultados de búsqueda
            _, _, similar_classes, distances, _ = self.evaluate(img, k=k)
            # Calcular relevancias
            relevances = [1 if (cls == true_label) else 0 for cls in similar_classes]

            # Calcular NDCG
            try:
                ndcg = ndcg_score(
                    [relevances],
                    [-np.array(distances)],  # Invertimos distancias
                    k=k,
                )
                ndcg_scores.append(ndcg)
            except Exception as e:
                print(f"Error calculating NDCG: {e}")
                ndcg_scores.append(0.0)

        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        return ndcg_scores, avg_ndcg


class DogSearchEngineV2(DogSearchEngine):
    def __init__(
        self,
        model_index: dict[str, tuple[Model, Index]],
        images: list[np.ndarray],
        labels: list[str],
        class_names: list[str],
    ):
        # Inicializar con el primer modelo del diccionario
        self.model_index = model_index
        self.model_name = "modelo_resnet50"  # Modelo predeterminado
        self.model = model_index[self.model_name][0]
        self.index = model_index[self.model_name][1]

        # Llamar al constructor de la clase base
        super().__init__(self.model, self.index, images, labels)

        # Atributos adicionales específicos de DogSearchEngineV2
        self.class_names = class_names

    def set_model(self, model_name: str):
        """
        Cambia el modelo activo.
        :param model_name: Nombre del modelo a usar.
        """
        if model_name not in self.model_index:
            raise ValueError(
                f"El modelo '{model_name}' no está disponible en el índice."
            )

        # Actualizar el modelo e índice activos
        self.model_name = model_name
        self.model = self.model_index[self.model_name][0]
        self.index = self.model_index[self.model_name][1]
