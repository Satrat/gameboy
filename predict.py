from cog import BasePredictor, Path, Input
import os
import torch
from uuid import uuid4
from PIL import Image
import torchvision.transforms as transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CHECKPOINT_PATH = "./checkpoint-epoch-10"

class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(CHECKPOINT_PATH):
            raise ValueError(f"Error: Model checkpoint {CHECKPOINT_PATH} not found!")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
        
        # Load the modified UNet (8 input channels)
        self.unet = UNet2DConditionModel.from_pretrained(
            CHECKPOINT_PATH,
            in_channels=8,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        ).to(self.device)
        
        # Set to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

    def preprocess_image(self, image_path):
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        print(f"Input image shape: {image.size}")
        
        # Same preprocessing as training
        preprocess = transforms.Compose([
            transforms.Resize((672, 768), interpolation=transforms.InterpolationMode.NEAREST), # upscale gameboy img to 6x resolution
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        # Apply transforms and add batch dimension
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        print(f"Input image modified shape: {image_tensor.shape}")
        return image_tensor

    def encode_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings

    def generate_image(
        self,
        input_image_path, 
        prompt="high quality colorized photograph, natural colors, detailed, full color",
        negative_prompt="",
        num_inference_steps=20,
        cfg_scale=7.5,
        seed=None
    ):
        if seed is not None:
            torch.manual_seed(seed)
        

        # Preprocess input image
        input_image = self.preprocess_image(input_image_path)
        print(f"Input image shape: {input_image.shape}")
        
        with torch.no_grad():
            # Encode input image to latents
            input_latents = self.vae.encode(input_image).latent_dist.sample() * self.vae.config.scaling_factor
            print(f"Input latents shape: {input_latents.shape}")
            
            # Encode text prompts
            text_embeddings = self.encode_text(prompt)
            uncond_embeddings = self.encode_text(negative_prompt)
            
            # Initialize random noise
            sample_latents = torch.randn_like(input_latents)
            
            # Set up scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            
            print(f"Starting generation with {num_inference_steps} steps...")
            
            # Denoising loop
            for i, timestep in enumerate(self.scheduler.timesteps):
                print(f"Step {i+1}/{num_inference_steps}")
                
                # Prepare UNet input (concatenate noisy latents with input latents)
                unet_input = torch.cat([sample_latents, input_latents], dim=1)
                print(f"UNet input shape: {unet_input.shape}")
                
                if cfg_scale > 1.0:
                    # Classifier-free guidance
                    unet_input_combined = torch.cat([unet_input, unet_input])
                    text_emb_combined = torch.cat([uncond_embeddings, text_embeddings])
                    timestep_batch = timestep.unsqueeze(0).repeat(2).to(self.device)
                    
                    # Single forward pass for both conditional and unconditional
                    noise_pred = self.unet(
                        unet_input_combined,
                        timestep_batch,
                        encoder_hidden_states=text_emb_combined
                    ).sample
                    
                    # Split and apply CFG
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    # No CFG
                    timestep_batch = timestep.unsqueeze(0).to(self.device)
                    noise_pred = self.unet(
                        unet_input,
                        timestep_batch,
                        encoder_hidden_states=text_embeddings
                    ).sample
                
                # Denoise step
                sample_latents = self.scheduler.step(noise_pred, timestep, sample_latents).prev_sample
            
            # Decode latents to image
            generated_image = self.vae.decode(sample_latents / self.vae.config.scaling_factor).sample
            
            # Convert to [0, 1] range
            generated_image = ((generated_image + 1) / 2).clamp(0, 1)
            generated_image = (generated_image * 255).clamp(0, 255).byte()
            
        return generated_image

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Text prompt for generation", 
            default="high quality colorized photograph, natural colors, detailed, full color"
        ),
        negative_prompt: str = Input(
            description="Negative prompt", 
            default="blurry, low quality"
        ),
        steps: int = Input(
            description="Number of inference steps", 
            default=20,
            ge=1,
            le=50
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale", 
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility. Leave blank for random seed",
            default=None
        )
    ) -> Path:
        if not os.path.exists(image):
            raise ValueError(f"Error: Input image {image} not found!")
        
        print(f"Prompt: {prompt}")
        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")
        
        generated_image = self.generate_image(
            image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            seed=seed
        )
        
        id = str(uuid4())
        output_path = f"{id}.jpg"
        generated_image_cpu = generated_image.cpu()
        generated_image_cpu = generated_image_cpu.squeeze(0).permute(1, 2, 0).numpy()
        print(f"Saving output of shape {generated_image.shape} to {id}")
        Image.fromarray(generated_image_cpu).save(output_path)
        return Path(output_path)