import numpy as np
import gradio as gr
from dogsearch import DogSearchEngine, DogSearchEngineV2
from dogdetection import DogDetectionClassificationPipeline


class GradioInterface:
    def __init__(self, search_engine, title: str, description: str):
        self.search_engine = search_engine
        self.title = title
        self.description = description

    def process_image_common(
        self, input_image: np.ndarray, model_name: str = None, k: int = 10
    ):
        if model_name:
            self.search_engine.set_model(model_name)

        input_img, similar_imgs, classes, distances, predicted_class = (
            self.search_engine.evaluate(input_image, k=k)
        )

        # Convertir imagen de entrada a uint8
        input_img = input_img.astype(np.uint8)

        # Preparar resultados para visualización
        results = [
            (img, f"{cls} (Sim: {100 * (1 - dist / max(distances)):.1f}%)")
            for img, cls, dist in zip(similar_imgs, classes, distances)
        ]

        return input_img, predicted_class, results

    def create_interface(
        self, inputs, outputs, process_fn, submit_label="Buscar Similares"
    ):
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# 🐶 {self.title}")
            gr.Markdown(self.description)

            with gr.Row():
                with gr.Column(scale=1):
                    for input_component in inputs:
                        input_component.render()

                    submit_btn = gr.Button(submit_label, variant="primary")

                with gr.Column(scale=2):
                    for output_component in outputs:
                        output_component.render()

            submit_btn.click(
                fn=process_fn,
                inputs=[input_component for input_component in inputs],
                outputs=[output_component for output_component in outputs],
            )

        return app

    def launch(
        self,
        inputs,
        outputs,
        process_fn,
        submit_label="Buscar Similares",
        share=True,
        debug=True,
    ):
        app = self.create_interface(inputs, outputs, process_fn, submit_label)
        app.launch(share=share, debug=debug)


class DogSearchInterface(GradioInterface):
    def __init__(self, search_engine: DogSearchEngine):
        super().__init__(
            search_engine=search_engine,
            title="Buscador de Razas de Perros",
            description="Sube una imagen de perro para encontrar las razas más similares en nuestra base de datos",
        )

    def process_image(self, input_image: np.ndarray):
        return self.process_image_common(input_image)

    def launch(self):
        inputs = [gr.Image(label="Imagen de entrada", type="numpy", height=300)]
        outputs = [
            gr.Image(label="Tu imagen", interactive=False, height=300),
            gr.Label(label="Raza predicha", container=True),
            gr.Gallery(
                label="Resultados (10 más similares)",
                columns=5,
                rows=2,
                object_fit="cover",
                height="auto",
            ),
        ]
        super().launch(inputs, outputs, self.process_image)


class DogSearchInterfaceV2(GradioInterface):
    def __init__(self, search_engine: DogSearchEngineV2):
        super().__init__(
            search_engine=search_engine,
            title="Buscador de Razas de Perros",
            description="Sube una imagen de perro para encontrar las razas más similares en nuestra base de datos",
        )
        self.model_choices = list(search_engine.model_index.keys())

    def process_image(self, input_image: np.ndarray, model_name: str):
        return self.process_image_common(input_image, model_name=model_name, k=10)

    def launch(self):
        inputs = [
            gr.Image(label="Imagen de entrada", type="numpy", height=300),
            gr.Dropdown(
                choices=self.model_choices,
                label="Selecciona un modelo",
                value=self.model_choices[2],
            ),
        ]
        outputs = [
            gr.Image(label="Tu imagen", interactive=False, height=300),
            gr.Label(label="Raza predicha", container=True),
            gr.Gallery(
                label="Resultados (10 más similares)",
                columns=5,
                rows=2,
                object_fit="cover",
                height="auto",
            ),
        ]
        super().launch(inputs, outputs, self.process_image)


class DogDetectionPipelineInterface(GradioInterface):
    def __init__(self, pipeline: DogDetectionClassificationPipeline):
        super().__init__(
            search_engine=None,
            title="Detección y Clasificación de Perros",
            description="Sube una imagen compleja para detectar y clasificar razas de perros",
        )
        self.pipeline = pipeline

    def process_image(self, input_image: np.ndarray):
        return self.pipeline.process_image(input_image)

    def launch(self):
        inputs = [
            gr.Image(label="Imagen de entrada", type="numpy", height=300),
        ]
        outputs = [
            gr.Image(label="Resultado de detección", interactive=False, height=400),
            gr.JSON(label="Detecciones"),
        ]
        super().launch(
            inputs, outputs, self.process_image, submit_label="Procesar Imagen"
        )
