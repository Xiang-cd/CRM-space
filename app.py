# Not ready to use yet
import argparse
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
import torch
from PIL import Image
import PIL
from pipelines import TwoStagePipeline
from huggingface_hub import hf_hub_download
import os
import rembg
from typing import Any

pipeline = None
rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def remove_background(
    image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def preprocess_image(image, do_remove_background, force_remove, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
    if do_remove_background:
        image = remove_background(image, rembg_session, force_remove)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")


def gen_image(input_image, seed, scale, step):
    global pipeline
    pipeline.set_seed(seed)
    rt_dict = pipeline(input_image, scale=scale, step=step)
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
            with gr.Row():
                image_input = gr.Image(
                    label="image input",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False, type="pil", image_mode="RGB")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                        force_remove = gr.Checkbox(label="Force Remove", value=False)
                    back_groud_color = gr.ColorPicker(label="Background Color")
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                    )

                with gr.Column():
                    seed = gr.Number(value=1234, label="seed", precision=0)
                    guidance_scale = gr.Number(value=5.5, label="guidance_scale")
                    step = gr.Number(value=50, label="sample steps", precision=0)
            text_button = gr.Button("Generate Images")
        with gr.Column():
            image_output = gr.Image(interactive=False, label="Output pixel image")
            xyz_ouput = gr.Image(interactive=False, label="Output xyz image")

            output_model = gr.Model3D(
                label="Output Model",
                interactive=False,
            )

    inputs = [
        processed_image,
        seed,
        guidance_scale,
        step,
    ]
    outputs = [
        image_output,
        xyz_ouput,
    ]
    gr.Examples(
        examples=[os.path.join("examples", i) for i in os.listdir("examples")],
        inputs=[image_input],
    )

    text_button.click(fn=check_input_image, inputs=[image_input]).success(
        fn=preprocess_image,
        inputs=[image_input, do_remove_background, force_remove, foreground_ratio, back_groud_color],
        outputs=[processed_image],
    ).success(
        fn=gen_image,
        inputs=inputs,
        outputs=outputs,
    )
    demo.queue().launch()
