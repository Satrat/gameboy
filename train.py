import os
from tqdm.auto import tqdm
from datasets import load_dataset
from dataclasses import dataclass

import torch
from torchvision import transforms
import torch.nn.functional as F 
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler 
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer

# config
device = "cuda:4" if torch.cuda.is_available() else "cpu" 
DATASET_PATH_HF = "Satrat/gameboy-faces"
OUTPUT_DIR = "stable-diffusion-v1-5-gameboy-upscaled-3"
FT_CHECKPOINT = "stable-diffusion-v1-5/stable-diffusion-v1-5"

@dataclass
class TrainingConfig:
    #image_width: int = 768 #128
    #image_height: int = 762 #112
    train_batch_size = 8
    eval_batch_size = 8
    num_epochs = 8
    gradient_accumulation_steps = 1
    learning_rate = 5e-6
    lr_warmup_steps = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'bf16' 
    output_dir = OUTPUT_DIR
    dataset_name = DATASET_PATH_HF

    push_to_hub = False
    hub_private_repo = True
    overwrite_output_dir = True
    seed = 0

config = TrainingConfig()

# preprocess data, create dataloaders
dataset_tr = load_dataset(config.dataset_name, split="train")
dataset_test = load_dataset(config.dataset_name, split="test")

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    in_imgs = []
    out_imgs = []
    
    for i in range(len(examples['input_image'])):
        img_in = examples["input_image"][i]
        img_out = examples["output_image"][i]

        input_img = preprocess(img_in.convert("RGB"))
        output_img = preprocess(img_out.convert("RGB"))
        in_imgs.append(input_img)
        out_imgs.append(output_img)
    
    return {"real_img": in_imgs, "gb_img": out_imgs}

dataset_tr.set_transform(transform)
dataset_test.set_transform(transform)

test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config.train_batch_size, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(dataset_tr, batch_size=config.train_batch_size, shuffle=True)


# Initialize model
vae = AutoencoderKL.from_pretrained(FT_CHECKPOINT, subfolder="vae").to(device)
scaling_factor = vae.config.scaling_factor
text_encoder = CLIPTextModel.from_pretrained(FT_CHECKPOINT, subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(FT_CHECKPOINT, subfolder="tokenizer")

model_id = FT_CHECKPOINT
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=8,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
)

original_unet = StableDiffusionPipeline.from_pretrained(model_id).unet
with torch.no_grad():
    # Copy original weights
    unet.conv_in.weight[:, :4] = original_unet.conv_in.weight

    torch.nn.init.kaiming_normal_(unet.conv_in.weight[:, 4:], mode='fan_out')
    unet.conv_in.weight[:, 4:] *= 0.1  # Scale down
unet = unet.to(device)

# Inference Test
sample = dataset_test[0]
gb_img = sample['gb_img'].unsqueeze(0).to(device)
real_img = sample['real_img'].unsqueeze(0).to(device)

with torch.no_grad():
    gb_latents = vae.encode(gb_img).latent_dist.sample() * scaling_factor
    real_latents = vae.encode(real_img).latent_dist.sample() * scaling_factor

# Add some noise
noise = torch.randn_like(real_latents)
noisy_latents = real_latents + 0.5 * noise

# Concatenate
unet_input = torch.cat([noisy_latents, gb_latents], dim=1)
print(unet_input.shape)

# Test UNet forward pass
timestep = torch.tensor([100], device=device)
with torch.no_grad():
    # empty text prompt
    text_emb = text_encoder(tokenizer("", return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device))[0]
    
    # Forward pass - this is the critical test
    noise_pred = unet(unet_input, timestep, encoder_hidden_states=text_emb).sample
    
print(f"Input: {unet_input.shape} -> Output: {noise_pred.shape}")

# Training Setup
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

# Optimizer and scheduler
optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay=0.01)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# Mixed precision setup
if config.mixed_precision == 'bf16':
    scaler = GradScaler()
    use_amp = True
else:
    use_amp = False

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# Training prompts
training_prompts = [
    "high quality colorized photograph, natural colors, detailed, full color",
    "photorealistic portrait, colorized, natural skin tones, enhanced details, colorful",
    "",
    "professional color photography, sharp focus, realistic, colorized, vibrant",
    "detailed color photograph, natural lighting, high resolution, vibrant colors",
    "",
]

# Training helper functions
def encode_text_batch(prompts):
    """Encode text prompts"""
    text_inputs = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        return text_encoder(text_inputs.input_ids.to(device))[0]

def get_text_embeddings(prompts, null_prob=0.15):
    """Get text embeddings with random unconditional prompts for CFG training"""
    # Randomly replace some prompts with empty string for CFG
    cfg_prompts = [p if torch.rand(1) > null_prob else "" for p in prompts]
    return encode_text_batch(cfg_prompts)

def generate_validation_samples(epoch, num_samples=4):
    """Enhanced validation with CFG, multiple prompts, and quality metrics"""
    print(f"Generating validation samples for epoch {epoch}...")
    unet.eval()
    
    # Diverse validation prompts
    val_prompts = training_prompts
    
    with torch.no_grad():
        # Get consistent validation batch (same samples each time for comparison)
        torch.manual_seed(42)  # Fixed seed for reproducible validation
        val_batch = next(iter(torch.utils.data.DataLoader(
            dataset_test, batch_size=num_samples, shuffle=True
        )))
        gb_images = val_batch['gb_img'].to(device)
        real_images = val_batch['real_img'].to(device)
        
        gb_latents = vae.encode(gb_images).latent_dist.sample() * scaling_factor
        
        inference_scheduler = noise_scheduler
        inference_scheduler.set_timesteps(20)
        
        all_generated = []
        
        # Generate with different prompts
        for prompt in val_prompts:
            sample_latents = torch.randn_like(gb_latents)
            
            # Prepare embeddings for CFG
            if prompt:
                text_embeddings = encode_text_batch([prompt] * num_samples)
            else:
                text_embeddings = encode_text_batch([""] * num_samples)
            
            uncond_embeddings = encode_text_batch([""] * num_samples)
            
            # CFG denoising loop
            for i, t in enumerate(inference_scheduler.timesteps):
                timestep = t.unsqueeze(0).repeat(num_samples).to(device)
                
                # Prepare input
                unet_input = torch.cat([sample_latents, gb_latents], dim=1)
                
                # Classifier-free guidance
                unet_input_combined = torch.cat([unet_input, unet_input])
                text_emb_combined = torch.cat([uncond_embeddings, text_embeddings])
                timestep_combined = torch.cat([timestep, timestep])
                
                # Single forward pass for both conditional and unconditional
                noise_pred = unet(
                    unet_input_combined, 
                    timestep_combined, 
                    encoder_hidden_states=text_emb_combined
                ).sample
                
                # Split and apply CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                cfg_scale = 3.5 if prompt else 1.0  # More conservative CFG scale
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
                
                # Denoise step
                sample_latents = inference_scheduler.step(noise_pred, t, sample_latents).prev_sample
            
            # Decode to images
            generated_images = vae.decode(sample_latents / scaling_factor).sample
            generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
            all_generated.append(generated_images)
        
        # Create comprehensive comparison grid
        gb_display = (gb_images * 0.5 + 0.5).clamp(0, 1)
        real_display = (real_images * 0.5 + 0.5).clamp(0, 1)
        
        # Layout: GB | Real | Gen1 | Gen2 | Gen3 | Gen4 (different prompts)
        comparison_rows = []
        for i in range(num_samples):
            row = [gb_display[i], real_display[i]] + [gen[i] for gen in all_generated]
            comparison_rows.append(torch.stack(row))
        
        # Stack all rows
        full_comparison = torch.cat(comparison_rows, dim=0)
        
        # Create grid with proper spacing
        grid = vutils.make_grid(
            full_comparison, 
            nrow=len(val_prompts) + 2,  # GB + Real + generated variants
            padding=4, 
            pad_value=1.0  # White padding
        )
        
        # Save main comparison
        vutils.save_image(grid, os.path.join(config.output_dir, f"validation_epoch_{epoch}.png"))
        
        # Save individual high-quality samples for detailed inspection
        for i, prompt in enumerate(val_prompts):
            individual_grid = vutils.make_grid(all_generated[i], nrow=2, padding=2)
            prompt_name = f"prompt_{i}" if prompt else "unconditional"
            vutils.save_image(
                individual_grid, 
                os.path.join(config.output_dir, f"epoch_{epoch}_{prompt_name}.png")
            )
    
    unet.train()
    print(f"Validation samples saved for epoch {epoch}")

def compute_validation_metrics(epoch, num_samples=16):
    """Compute quantitative metrics on validation set"""
    unet.eval()
    
    with torch.no_grad():
        val_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=4, shuffle=False
        )
        
        total_mse = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx * 4 >= num_samples:
                break
                
            gb_images = batch['gb_img'].to(device)
            real_images = batch['real_img'].to(device)
            
            # Quick generation (fewer steps for speed)
            gb_latents = vae.encode(gb_images).latent_dist.sample() * scaling_factor
            sample_latents = torch.randn_like(gb_latents)
            
            # Simplified generation for metrics
            inference_scheduler = noise_scheduler
            inference_scheduler.set_timesteps(20)
            
            text_embeddings = encode_text_batch(
                ["colorized photograph, natural colors"] * gb_images.shape[0]
            )
            
            for t in inference_scheduler.timesteps:
                timestep = t.unsqueeze(0).repeat(gb_images.shape[0]).to(device)
                unet_input = torch.cat([sample_latents, gb_latents], dim=1)
                noise_pred = unet(unet_input, timestep, encoder_hidden_states=text_embeddings).sample
                sample_latents = inference_scheduler.step(noise_pred, t, sample_latents).prev_sample
            
            # Decode and compute metrics
            generated_images = vae.decode(sample_latents / scaling_factor).sample
            generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
            real_images_norm = (real_images * 0.5 + 0.5).clamp(0, 1)
            
            # MSE in pixel space
            mse = F.mse_loss(generated_images, real_images_norm)
            total_mse += mse.item() * gb_images.shape[0]
            total_samples += gb_images.shape[0]
        
        avg_mse = total_mse / total_samples
        print(f"Epoch {epoch} - Validation MSE: {avg_mse:.6f}")
        
        # Log to a file for tracking
        with open(os.path.join(config.output_dir, "validation_metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch}: MSE={avg_mse:.6f}\n")
    
    unet.train()

# Training loop
print("Starting training...")
global_step = 0
all_losses = []  # Track losses for monitoring

generate_validation_samples(0)
compute_validation_metrics(0)

for epoch in range(config.num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
    
    for step, batch in enumerate(progress_bar):
        # Get batch data
        gb_images = batch['gb_img'].to(device)
        real_images = batch['real_img'].to(device)
        batch_size = gb_images.shape[0]
        
        # Encode to latents
        with torch.no_grad():
            gb_latents = vae.encode(gb_images).latent_dist.sample() * scaling_factor
            real_latents = vae.encode(real_images).latent_dist.sample() * scaling_factor
        
        # Add noise
        noise = torch.randn_like(real_latents)
        timesteps = torch.randint(1, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
        noisy_real_latents = noise_scheduler.add_noise(real_latents, noise, timesteps)
        
        # Get text embeddings with CFG training
        prompts = [training_prompts[i % len(training_prompts)] for i in range(batch_size)]
        text_embeddings = get_text_embeddings(prompts)  # Uses CFG null probability
        unet_input = torch.cat([noisy_real_latents, gb_latents], dim=1)
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                noise_pred = unet(unet_input, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        else:
            noise_pred = unet(unet_input, timesteps, encoder_hidden_states=text_embeddings).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
        
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Update progress and tracking
        epoch_loss += loss.item()
        all_losses.append(loss.item())
        global_step += 1
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

    # Epoch summary
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")
    
    # Save checkpoints and samples
    if (epoch + 1) % config.save_model_epochs == 0:
        save_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch + 1}")
        unet.save_pretrained(save_path)
        print(f"Saved: {save_path}")
    
    # Enhanced validation with CFG and multiple prompts
    if (epoch + 1) % config.save_image_epochs == 0:
        generate_validation_samples(epoch + 1)
        compute_validation_metrics(epoch + 1)

# Final save
unet.save_pretrained(os.path.join(config.output_dir, "final_model"))
print("Training complete!")
