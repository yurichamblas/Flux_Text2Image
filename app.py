import gradio as gr
import spaces
from diffusers import DiffusionPipeline
import torch
import os
import random
from huggingface_hub import login

# Autenticarse en HuggingFace con el token de acceso
login(os.getenv("HUGGINGFACEHUB_TOKEN"))

# Asegurarse de que SentencePiece esté instalado
try:
    import sentencepiece
except ImportError:
    os.system("pip install sentencepiece")

# Inicializar el pipeline Diffusers
model_id = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEED = 2**32 - 1
MAX_IMAGE_SIZE = 1024

@spaces.GPU
def generate_image(prompt, seed, random_seed, width, height, guidance_scale, num_inference_steps):
    # Controlar el seed para reproducibilidad
    if random_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.manual_seed(seed)

    # Generar la imagen con parámetros de Diffusers
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return image, seed

examples = [
    ["An illustration of a School of Architecture building surrounded by people in an urban public space"],
    ["A high-quality architectural photograph of an office building, in an urban context"],
    ["A residential courtyard featuring landscaped playgrounds and seating areas in an urban environment"],
    ["A high-quality architectural photograph of a mixed-use high-rise building with a colorful façade"]
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# AI Studio")

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Describe la imagen que deseas...",
                container=False,
            )

            run_button = gr.Button("Generar", scale=0, variant="primary")

        result = gr.Image(label="Resultado", show_label=False)

        with gr.Accordion("Configuración Avanzada", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            random_seed = gr.Checkbox(label="Randomizar seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Ancho",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Alto",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=15.0,
                    step=0.1,
                    value=7.5,
                )

                num_inference_steps = gr.Slider(
                    label="Número de pasos de inferencia",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=20,
                )

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        generate_image,
        inputs=[
            prompt,
            seed,
            random_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()