# 🐶 TP Final Computer Vision: Reconocimiento y Detección de Razas de Perros

**Autor:** Bravi Eugenio B-6600/1

## 📄 Resumen del Proyecto

Este proyecto es un Trabajo Práctico Final de Computer Vision que aborda el desafío de la detección y clasificación de razas de perros en imágenes. Se construye una solución integral que abarca desde la preparación del dataset y la implementación de bases de datos vectoriales para búsqueda por similitud, hasta el entrenamiento de modelos de clasificación profunda y la creación de un pipeline completo de detección + clasificación, culminando con la optimización de modelos y un script de anotación automática.

Las principales tecnologías utilizadas incluyen:
*   **TensorFlow/Keras**: Para la construcción y entrenamiento de modelos de Deep Learning.
*   **FAISS**: Para la creación de una base de datos vectorial eficiente para la búsqueda por similitud de imágenes.
*   **Ultralytics YOLOv8**: Para la detección de objetos (perros) en imágenes complejas.
*   **Gradio**: Para la creación de interfaces de usuario interactivas y demostrativas.
*   **KaggleHub**: Para la descarga del dataset.
*   **OpenCV (cv2)**: Para el preprocesamiento y manipulación de imágenes.
*   **scikit-learn**: Para el cálculo de métricas de evaluación.
*   **TensorFlow Lite**: Para la optimización y cuantización de modelos.

## 🚀 Etapas del Proyecto

El proyecto se divide en varias etapas clave:

### 1. Preparación del Dataset y Base de Datos Vectorial para Búsqueda por Similitud

*   **Descarga y Análisis del Dataset**: El dataset `70-dog-breedsimage-data-set` se descarga de Kaggle. Se realiza un análisis inicial que revela un desequilibrio significativo en la distribución de las clases (razas de perros).
*   **Balanceo del Dataset con Data Augmentation**: Para mitigar el desbalance, se implementa una estrategia de aumento de datos (`data_augmentation`) que genera nuevas imágenes para las clases minoritarias, utilizando transformaciones como volteo horizontal/vertical, rotación, contraste y brillo.
*   **Base de Datos Vectorial (FAISS)**:
    *   Se utiliza el modelo pre-entrenado **ResNet50** (sin la capa superior) como extractor de características para generar embeddings (representaciones vectoriales) de las imágenes.
    *   Estos embeddings se indexan utilizando **FAISS (Facebook AI Similarity Search)**, permitiendo búsquedas eficientes de imágenes similares basadas en la distancia euclidiana (L2).
*   **Buscador por Similitud (Gradio)**: Se desarrolla una interfaz interactiva con Gradio que permite al usuario subir una imagen y encontrar las 10 razas de perros más similares en la base de datos, mostrando la imagen de entrada, la raza predicha por voto mayoritario y las imágenes y clases similares.
*   **Evaluación del Buscador**: Se calcula el **NDCG@10 (Normalized Discounted Cumulative Gain)** para evaluar la calidad del ranking de similitud. Se observa una mejora significativa en el NDCG@10 (de ~0.78 a ~0.83) después de aplicar el aumento de datos, demostrando la importancia de un dataset balanceado.

### 2. Entrenamiento y Evaluación de Modelos de Clasificación

*   **Pipeline de Entrenamiento y Evaluación**: Se implementa una función `train_test` que gestiona el preprocesamiento de etiquetas (one-hot encoding), la creación de `tf.data.Dataset`, la compilación del modelo (optimizador Adam, `categorical_crossentropy`), y el uso de callbacks avanzados (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint). La función también visualiza el historial de entrenamiento (accuracy y loss) y genera un reporte de clasificación completo (precision, recall, f1-score) y visualizaciones de predicciones.
*   **Modelos Entrenados**:
    *   **Modelo Convolucional (desde cero)**: Una red neuronal convolucional personalizada. Este modelo mostró un rendimiento modesto (accuracy de prueba de ~0.46), lo que resalta la complejidad de la tarea sin un pre-entrenamiento.
    *   **ResNet50 con Transfer Learning (Finetuning)**: Se utiliza ResNet50 pre-entrenado en ImageNet como extractor de características, seguido de capas densas personalizadas. Este modelo alcanzó un rendimiento significativamente superior (accuracy de prueba de ~0.88), demostrando la eficacia del transfer learning para esta tarea.
*   **Carga de Modelos y Nuevos Índices FAISS**: Se cargan los pesos de los modelos entrenados y se generan nuevos índices FAISS utilizando las capas de embedding (`densa_embbeding`) de estos modelos, permitiendo comparar su rendimiento en la búsqueda por similitud.
*   **Buscador por Similitud V2 (Gradio)**: Se mejora la interfaz de Gradio para permitir al usuario seleccionar entre los diferentes modelos (ResNet50 original, ResNet50 con finetuning, CNN desde cero) para realizar la búsqueda por similitud.
*   **Comparación de Modelos en Búsqueda por Similitud**: Se evalúa el NDCG@10 para cada uno de los modelos entrenados como extractores de características para la búsqueda por similitud. Se observa que el modelo ResNet50 con finetuning mantiene un alto rendimiento, ligeramente inferior al ResNet50 base en esta métrica.

### 3. Pipeline de Detección y Clasificación (YOLO + Clasificador)

*   **Integración de Detección y Clasificación**: Se crea un `DogDetectionClassificationPipeline` que combina:
    *   **Detección de objetos con YOLOv8n**: Se utiliza un modelo YOLOv8n pre-entrenado para identificar y localizar perros en imágenes.
    *   **Clasificación de razas**: La porción de perro detectada por YOLO es recortada y pasada al modelo de clasificación de razas (el `modelo_transfer_learning` previamente entrenado) para determinar su raza.
*   **Interfaz Gradio para el Pipeline Completo**: Se implementa una interfaz de usuario en Gradio que permite al usuario subir una imagen compleja, y el pipeline detecta los perros, clasifica sus razas y muestra la imagen anotada con los bounding boxes y las etiquetas de raza, junto con un JSON de las detecciones.

### 4. Evaluación del Pipeline Completo y Optimización

*   **Anotación de Imágenes Complejas**: Se descargan y anotan manualmente un conjunto de imágenes "complejas" (con múltiples perros, o perros en diversos contextos) para servir como Ground Truth para la evaluación del pipeline completo. Se utiliza `cv2` para visualizar estas anotaciones.
*   **Evaluación del Pipeline**: Se define una función `evaluate_pipeline` para medir el rendimiento del pipeline de detección y clasificación. Esta función calcula métricas clave como **Precision, Recall, F1-Score, Mean IoU (Intersection over Union)** para la detección, y un **mAP@0.5** (Mean Average Precision a un umbral IoU de 0.5) para evaluar la combinación de detección y clasificación.
    *   Los resultados obtenidos (Precision: ~0.76, Recall: ~0.62, F1-Score: ~0.67, Mean IoU: ~0.87, mAP@0.5: ~0.65) indican un buen rendimiento en la clasificación de las detecciones correctas y calidad de las bounding boxes, pero una limitación notable en el `Recall` de la detección (es decir, no detecta todos los perros presentes).
*   **Optimización de Modelos con TensorFlow Lite**: El `modelo_transfer_learning` se cuantiza a formato **INT8** utilizando TensorFlow Lite. Esto reduce significativamente el tamaño del modelo y acelera la inferencia.
    *   Se implementa una clase `classify_image_tflite` para medir el tiempo de inferencia del modelo optimizado.
    *   La evaluación del pipeline con el modelo optimizado muestra métricas de rendimiento idénticas al modelo original, pero con una **velocidad de inferencia aproximadamente el doble de rápida** (de ~40ms a ~20ms), confirmando la eficacia de la cuantización.
*   **Script de Anotación Automática**: Se desarrolla un script (`script_anotacion`) que utiliza el pipeline de detección y clasificación para procesar automáticamente una carpeta de imágenes, generar anotaciones en formatos **YOLO** (`.txt`) y **COCO** (`.json`), y opcionalmente guardar las imágenes con las anotaciones dibujadas. Este script es útil para automatizar la creación de datasets anotados.

## ⚙️ Cómo Ejecutar el Proyecto

Este proyecto está diseñado para ejecutarse en un entorno como Google Colab debido a las dependencias de GPU para el entrenamiento y la inferencia de modelos de Deep Learning.

1.  **Abrir en Google Colab**: Sube el archivo `TP_FINAL_CV.ipynb` a Google Colab.
2.  **Ejecutar Celdas**: Ejecuta cada celda del notebook secuencialmente.
    *   La primera vez que ejecutes, se instalarán las dependencias (`kagglehub`, `faiss-cpu`, `gdown`, `ultralytics`).
    *   Se descargarán los datasets y los modelos pre-entrenados/guardados.
    *   Las interfaces de Gradio se lanzarán y proporcionarán enlaces públicos para su interacción.
