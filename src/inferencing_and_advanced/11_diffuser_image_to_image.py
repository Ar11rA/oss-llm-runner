"""
Image-to-Image Generation Examples

This file demonstrates image editing with diffusion models:
1. InstructPix2Pix - Edit images with text instructions (lightweight)
2. SDXL Img2Img - Higher quality image transformation
3. FLUX Kontext - State-of-the-art editing (requires 80GB+ VRAM)

Hardware Requirements:
- InstructPix2Pix: ~6-8 GB VRAM
- SDXL Img2Img: ~12-15 GB VRAM  
- FLUX Kontext: ~40-80 GB VRAM
"""

import torch
from PIL import Image

# =============================================================================
# EXAMPLE 1: InstructPix2Pix (Best for low VRAM, ~6GB)
# Edit images using natural language instructions
# =============================================================================

def instruct_pix2pix_example(image_path: str, instruction: str, output_path: str = "edited.png"):
    """Edit an image using text instructions."""
    from diffusers import StableDiffusionInstructPix2PixPipeline
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    
    # Load local image with PIL
    image = Image.open(image_path).convert("RGB")
    
    edited = pipe(
        prompt=instruction,
        image=image,
        num_inference_steps=20,
        image_guidance_scale=1.5,  # How much to follow original image
        guidance_scale=7.5,        # How much to follow text
    ).images[0]
    
    edited.save(output_path)
    print(f"Saved to {output_path}")
    return edited


# =============================================================================
# EXAMPLE 2: SDXL Image-to-Image (Higher quality, ~12GB)
# Transform images with style/content changes
# =============================================================================

def sdxl_img2img_example(image_path: str, prompt: str, strength: float = 0.5, output_path: str = "transformed.png"):
    """Transform an image using SDXL.
    
    Args:
        strength: 0.0 = no change, 1.0 = complete regeneration
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    
    # Load local image with PIL
    image = Image.open(image_path).convert("RGB")
    
    edited = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=30,
    ).images[0]
    
    edited.save(output_path)
    print(f"Saved to {output_path}")
    return edited


# =============================================================================
# EXAMPLE 3: FLUX Kontext (State-of-the-art, requires 80GB or offloading)
# Best quality image editing with text instructions
# =============================================================================

def flux_kontext_example(image_path: str, instruction: str, output_path: str = "flux_edited.png"):
    """Edit image with FLUX Kontext (requires large GPU or CPU offloading)."""
    from diffusers import FluxKontextPipeline
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
    )
    # Use CPU offloading for GPUs < 80GB
    pipe.enable_sequential_cpu_offload()
    
    # Load local image with PIL
    image = Image.open(image_path).convert("RGB")
    
    edited = pipe(
        image=image,
        prompt=instruction,
        guidance_scale=2.5,
    ).images[0]
    
    edited.save(output_path)
    print(f"Saved to {output_path}")
    return edited


# =============================================================================
# MAIN: Run examples
# =============================================================================

if __name__ == "__main__":
    # Example usage - uncomment the one you want to use
    
    # InstructPix2Pix (lightweight, ~6GB VRAM)
    # instruct_pix2pix_example(
    #     image_path="/content/bar1.jpg",
    #     instruction="Add wizard hats to the people",
    #     output_path="wizard_hats.png"
    # )
    
    # SDXL Img2Img (medium, ~12GB VRAM)
    # sdxl_img2img_example(
    #     image_path="/content/bar1.jpg",
    #     prompt="A fantasy tavern with magical lighting, medieval style",
    #     strength=0.6,
    #     output_path="fantasy_bar.png"
    # )
    
    # FLUX Kontext (heavy, ~40-80GB VRAM)
    # flux_kontext_example(
    #     image_path="/content/bar1.jpg",
    #     instruction="Add wizard hats to the people",
    #     output_path="flux_wizard.png"
    # )
    
    print("Uncomment an example above to run!")