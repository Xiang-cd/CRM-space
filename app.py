# Not ready to use yet
import random
import argparse
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
import torch
from PIL import Image
from pipelines import TwoStagePipeline
from huggingface_hub import hf_hub_download

pipeline = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_image(input_image, seed, scale, step):
    global pipeline
    pipeline.set_seed(seed)
    rt_dict = pipeline(Image.fromarray(input_image), scale=scale, step=step)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    np_xyzs = np.concatenate(stage2_images, 1)

    return Image.fromarray(np_imgs), Image.fromarray(np_xyzs)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--stage1_config",
    type=str,
    default="configs/nf7_v3_SNR_rd_size_stroke.yaml",
    help="config for stage1",
)
parser.add_argument(
    "--stage2_config",
    type=str,
    default="configs/stage2-v2-snr.yaml",
    help="config for stage2",
)

parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

stage1_config = OmegaConf.load(args.stage1_config).config
stage2_config = OmegaConf.load(args.stage2_config).config
stage2_sampler_config = stage2_config.sampler
stage1_sampler_config = stage1_config.sampler

stage1_model_config = stage1_config.models
stage2_model_config = stage2_config.models

xyz_path = hf_hub_download(repo_id="Xiang-cd/test-6view", filename="xyz.pth")
pixel_path = hf_hub_download(repo_id="Xiang-cd/test-6view", filename="pixel.pth")
stage1_model_config.resume = pixel_path
stage2_model_config.resume = xyz_path

pipeline = TwoStagePipeline(
    stage1_model_config,
    stage2_model_config,
    stage1_sampler_config,
    stage2_sampler_config,
    device=args.device,
    dtype=torch.float16
)

with gr.Blocks() as demo:
    gr.Markdown("# imagedream demo for multi-view images generation from single image")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="image input")
            seed = gr.Number(value=1234, label="seed", precision=0)
            guidance_scale = gr.Number(value=5.5, label="guidance_scale")
            step = gr.Number(value=50, label="sample steps", precision=0)
            text_button = gr.Button("Generate Images")
        with gr.Column():
            image_output = gr.Image(interactive=False)
            xyz_ouput = gr.Image(interactive=False)

            output_model = gr.Model3D(
                    label="Output Model",
                    interactive=False,
            )

    inputs = [
        image_input,
        seed,
        guidance_scale,
        step,
    ]
    outputs = [
        image_output,
        xyz_ouput,
    ]
    default_params = [1234, 5.5, 50]
    # gr.Examples(
    #     [ [i] + default_params for i in
    #         [
    #             "examples/3D卡通狗.png",
    #             "examples/大头泡泡马特.png"
    #         ]
    #     ],
    #     inputs=inputs,
    # )

    text_button.click(gen_image, inputs=inputs, outputs=outputs)
    demo.queue().launch()
