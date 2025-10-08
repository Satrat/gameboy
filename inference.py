import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def load_model(checkpoint_path, base_model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    print(f"Loading model from {checkpoint_path}")
    
    # Load base components
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    
    # Load the modified UNet (8 input channels)
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint_path,
        in_channels=8,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    ).to(device)
    
    # Set to eval mode
    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    return {
        'vae': vae,
        'text_encoder': text_encoder,
        'tokenizer': tokenizer,
        'unet': unet,
        'scheduler': scheduler
    }

def preprocess_image(image_path, device):
    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    
    # Same preprocessing as training
    preprocess = transforms.Compose([
        transforms.Resize((672, 768), interpolation=transforms.InterpolationMode.NEAREST), # upscale gameboy img to 6x resolution
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor

def encode_text(prompt, tokenizer, text_encoder, device):
    text_inputs = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    return text_embeddings

def generate_image(
    input_image_path, 
    models, 
    prompt="high quality colorized photograph, natural colors, detailed, full color",
    negative_prompt="",
    num_inference_steps=20,
    cfg_scale=7.5,
    device="cuda",
    seed=None
):
    if seed is not None:
        torch.manual_seed(seed)
    
    vae = models['vae']
    text_encoder = models['text_encoder']
    tokenizer = models['tokenizer']
    unet = models['unet']
    scheduler = models['scheduler']
    
    # Preprocess input image
    input_image = preprocess_image(input_image_path, device)
    input_copy_to_save = input_image.detach().clone() 
    print(f"Input image shape: {input_image.shape}")
    
    with torch.no_grad():
        # Encode input image to latents
        input_latents = vae.encode(input_image).latent_dist.sample() * vae.config.scaling_factor
        print(f"Input latents shape: {input_latents.shape}")
        
        # Encode text prompts
        text_embeddings = encode_text(prompt, tokenizer, text_encoder, device)
        uncond_embeddings = encode_text(negative_prompt, tokenizer, text_encoder, device)
        
        # Initialize random noise
        sample_latents = torch.randn_like(input_latents)
        
        # Set up scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        print(f"Starting generation with {num_inference_steps} steps...")
        
        # Denoising loop
        for i, timestep in enumerate(scheduler.timesteps):
            print(f"Step {i+1}/{num_inference_steps}")
            
            # Prepare UNet input (concatenate noisy latents with input latents)
            unet_input = torch.cat([sample_latents, input_latents], dim=1)
            print(f"UNet input shape: {unet_input.shape}")
            
            if cfg_scale > 1.0:
                # Classifier-free guidance
                unet_input_combined = torch.cat([unet_input, unet_input])
                text_emb_combined = torch.cat([uncond_embeddings, text_embeddings])
                timestep_batch = timestep.unsqueeze(0).repeat(2).to(device)
                
                # Single forward pass for both conditional and unconditional
                noise_pred = unet(
                    unet_input_combined,
                    timestep_batch,
                    encoder_hidden_states=text_emb_combined
                ).sample
                
                # Split and apply CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            else:
                # No CFG
                timestep_batch = timestep.unsqueeze(0).to(device)
                noise_pred = unet(
                    unet_input,
                    timestep_batch,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Denoise step
            sample_latents = scheduler.step(noise_pred, timestep, sample_latents).prev_sample
        
        # Decode latents to image
        generated_image = vae.decode(sample_latents / vae.config.scaling_factor).sample
        
        # Convert to iunt8
        generated_image = ((generated_image + 1) / 2).clamp(0, 1)
        generated_image = (generated_image * 255).clamp(0, 255).byte()
        input_copy_to_save = ((input_copy_to_save + 1) / 2).clamp(0, 1)
        input_copy_to_save = (input_copy_to_save * 255).clamp(0, 255).byte()

    return input_copy_to_save, generated_image

def save_comparison(input_path, input_tensor, generated_tensor, output_path):
    # Move generated tensor to CPU for saving
    generated_tensor_cpu = generated_tensor.cpu()
    input_tensor_cpu = input_tensor.cpu()
    print(generated_tensor.shape, input_tensor_cpu.shape)


    generated_tensor_cpu = generated_tensor_cpu.squeeze(0).permute(1, 2, 0).numpy()
    input_tensor_cpu = input_tensor_cpu.squeeze(0).permute(1, 2, 0).numpy()


    Image.fromarray(generated_tensor_cpu).save("inferred_output.jpg")
    Image.fromarray(input_tensor_cpu).save("inferred_input.jpg")



def main():
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned diffusion model")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", default="output.png", help="Output image path")
    parser.add_argument("--prompt", "-p", 
                       default="high quality colorized photograph, natural colors, detailed",
                       help="Text prompt for generation")
    parser.add_argument("--negative_prompt", "-n", default="blurry, low quality", 
                       help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, 
                       help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, 
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5",
                       help="Base model identifier")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    try:
        models = load_model(args.checkpoint, args.base_model, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate image
    try:
        print(f"Generating image from {args.input}...")
        print(f"Prompt: {args.prompt}")
        if args.negative_prompt:
            print(f"Negative prompt: {args.negative_prompt}")
        
        input_image, generated_image = generate_image(
            args.input,
            models,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            device=device,
            seed=args.seed
        )
        
        # Save results
        save_comparison(args.input, input_image, generated_image, args.output)
        print(f"Results saved to {args.output}")
        print(f"Generated image only saved to {args.output.replace('.png', '_generated_only.png')}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()