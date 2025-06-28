# 帮我写一个最简单版本的lumina_next_t2i_inference.py的代码，使用diffusers库中的LuminaPipeline类，并且能够处理输入参数，包括prompt、output路径、图像尺寸等。确保代码能够在GPU上运行，并且能够保存生成的图像到指定路径。
import argparse
import torch
from diffusers import LuminaPipeline

def lumina_inf(args):
    # Load the Lumina model
    pipeline = LuminaPipeline.from_pretrained(
        "/nfs_ssd/model/OriginModel/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16
    ).to("cuda")

    # Generate the image
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    image = pipeline(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
    ).images[0]
    image.save(args.output)
    print(f"Image saved to {args.output}")
if __name__ == "__main__":
    # Save the generated image
    parser = argparse.ArgumentParser(description="Lumina Next T2I Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--output", type=str, required=True, help="Output path for the generated image")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    lumina_inf(args)
    
    

