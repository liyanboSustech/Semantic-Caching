# 帮我写一个调用stable-diffusionmodel的python脚本，输入一个prompt，输出一张图片
import argparse
import torch
from diffusers import StableDiffusion3Pipeline

def generate_image(prompt, model_id="/nfs_ssd/model/OriginModel/stable-diffusion-3-medium-diffusers", output_path="output.png"):
    # Load the pre-trained Stable Diffusion model
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Generate the image
    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Save the generated image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    prompt = input("Enter a prompt for image generation: ")
    generate_image(prompt)
    