import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

OUTPUT_FOLDER = Path(__file__).parent.parent / "data" / "inference"

# Good quality ALP!
lora_path = Path(
    # Good quality ALP.
    # "/home/wizard/repositories/ai-toolkit/output/alp_flux_2024_08_22_09_10/alp_flux_2024_08_22_09_10.safetensors"
    # Better quality ALP, also with simpler training.
    # "/home/wizard/repositories/ai-toolkit/output/alp_flux_2024_08_23_14_10/alp_flux_2024_08_23_14_10_000002500.safetensors"
    # Testing.
    "/home/wizard/repositories/bendai/research/text_to_image_demo/data/training/alp_10_cropped_2024-08-25-06-58-42/checkpoints/alp_10_cropped_2000.safetensors"
)
# lora_path = Path(
#     # Good quality Rebi.
#     "/home/wizard/repositories/ai-toolkit/output/rebi_flux_2024_08_22_19_10/rebi_flux_2024_08_22_19_10.safetensors"
# )
adapter_name = "alp"
gender: Optional[str] = "man"
prompts: list[str] = [
    "A photo of <dvdpgl> holding a cute caramel long-fur dachshund, gentle lighting, pastel colors, low contrast, bokeh background. Both <dvdpgl> and the dog are smiling.",
    "Minimalist photography portrait of <dvdpgl>, minimalist, wearing a tiger hip hop bomber jacket and a t-shirt, purple background, low saturation.",
    "A high-key photo of <dvdpgl> dressed as a colorful clown amd holding an inflatable banana. Flash photography, smiling, with glasses.",
    "A photo of <dvdpgl> as a homeless person in the New York suburbs. Scrappy neighbourhood in the background, slightly blurry. A barrel on fire. Dirty image, scruffy, gritty, grunge, soft focus.",
    "<dvdpgl> as a hair-metal guitarist from the 80s. Playing a flying V guitar, long hair, leather jacket, tight pants, high boots, sunglasses. Stage lights in the background.",
    "<dvdpgl> with a flying squirrel lying on his head. Dressed in modern Japanese street fashion. Soft lighting, traditional Tokyo lights and street in the background. Slight bokeh effect.",
    "A pretty glossy picture of <dvdpgl> dressed as a Victorian princess. Royal pink clothes, a tiara on the head, a fan in the hand. Soft lighting, pastel colors. Inside Versailles. On the wall, a painting of a dachshund with a crown on its head.",
]
number_of_images_per_batch = 2
number_of_batches = 1
seed = 45
# It takes ~200 seconds to compile, and it saves ~2.5 seconds per image.
# Turn it on if you need to generate a lot of images and you can offset the compilation time.
compile = False

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
    )
    pipeline.load_lora_weights(
        pretrained_model_name_or_path_or_dict=str(lora_path),
        adapter_name=adapter_name,
    )
    pipeline.fuse_lora(adapter_names=[adapter_name])
    pipeline.to("cuda")
    assert isinstance(pipeline, FluxPipeline)

    if compile:
        pipeline.transformer = torch.compile(
            pipeline.transformer, mode="reduce-overhead", fullgraph=True
        )

    output_folder = OUTPUT_FOLDER / (
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{lora_path.stem}"
    )
    output_folder.mkdir(exist_ok=True, parents=True)
    print(f"💾 saving images to {output_folder}")

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    print(
        f"🔥 generating {number_of_images_per_batch} images per batch, {number_of_batches} batches"
    )

    if gender is not None:
        prompts = [p + f" <dvdpgl> is a {gender}." for p in prompts]

    with torch.inference_mode():
        for prompt_i, prompt in enumerate(prompts):
            print(f"👉 {prompt_i + 1}/{len(prompts)}) {prompt}")
            for iteration in range(number_of_batches):
                start = time.time()
                images = pipeline(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=40,
                    guidance_scale=3.5,
                    num_images_per_prompt=number_of_images_per_batch,
                    generator=generator,
                ).images  # type: ignore
                end = time.time()
                print(
                    f"✅ {len(images)} images generated in {end - start:.2f} seconds, {(end - start) / len(images):.2f} sec/image"
                )
                prompt_str = prompt.lower().replace(" ", "_")
                for image_i, image in enumerate(images):
                    image.save(
                        output_folder
                        / f"flux_lora_{prompt_i}_{prompt_str[:100]}_{iteration}_{image_i}.jpeg"
                    )
                torch.cuda.empty_cache()
