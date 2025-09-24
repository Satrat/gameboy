# %% [markdown]
# ## Check and Prep Dataset for Training

# %%
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass
import torch

# %%
@dataclass
class TrainingConfig:
    image_width: int = 768 #128
    image_height: int = 672 #112 
    train_batch_size = 8
    eval_batch_size = 8
    num_epochs = 4
    gradient_accumulation_steps = 1
    learning_rate = 5e-6
    lr_warmup_steps = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'bf16' 
    output_dir = 'stable-diffusion-v1-5-gameboy-upscaled'

    push_to_hub = False
    hub_private_repo = True
    overwrite_output_dir = True
    seed = 0

config = TrainingConfig()

# %%
config.dataset_name = "Satrat/gameboy-faces"
dataset_tr = load_dataset(config.dataset_name, split="train")
dataset_test = load_dataset(config.dataset_name, split="test")

# %%
num_ex = 5
fig, axs = plt.subplots(num_ex, 2, figsize=(4, 8)) 
for i in range(num_ex):
    stuff = dataset_test[i]
    img_in = stuff["input_image"]
    img_out = stuff["output_image"]
    
    axs[i, 0].imshow(img_in) 
    axs[i, 0].set_axis_off()
    axs[i, 1].imshow(img_out) 
    axs[i, 1].set_axis_off()
    
plt.tight_layout()
plt.show()

# %%
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
        print("PREPROCESS", input_img.shape, output_img.shape)
        in_imgs.append(input_img)
        out_imgs.append(output_img)
    
    return {"real_img": in_imgs, "gb_img": out_imgs}

dataset_tr.set_transform(transform)
dataset_test.set_transform(transform)

# %%
num_ex = 5
fig, axs = plt.subplots(num_ex, 2, figsize=(4, 8)) 
for i in range(num_ex):
    stuff = dataset_test[i]
    input_tensor = stuff["real_img"]
    output_tensor = stuff["gb_img"]

    input_img = (input_tensor * 0.5 + 0.5).clamp(0, 1)
    output_img = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Convert to numpy and transpose for matplotlib (C, H, W) -> (H, W, C)
    input_np = input_img.permute(1, 2, 0).numpy()
    output_np = output_img.permute(1, 2, 0).numpy()
    
    axs[i, 0].imshow(input_np) 
    axs[i, 0].set_axis_off()
    axs[i, 1].imshow(output_np) 
    axs[i, 1].set_axis_off()
    
plt.tight_layout()
plt.show()

# %%
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config.train_batch_size, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(dataset_tr, batch_size=config.train_batch_size, shuffle=True)

# %% [markdown]
# ## Test Shapes and Inference

# %%
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F 
import torch

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").to(device)
scaling_factor = vae.config.scaling_factor
text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")

# %%
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=8,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
)

# %%
original_unet = StableDiffusionPipeline.from_pretrained(model_id).unet
with torch.no_grad():
    # Copy original weights
    unet.conv_in.weight[:, :4] = original_unet.conv_in.weight

    # Initialize new channels with small random values instead of scaled copies
    torch.nn.init.kaiming_normal_(unet.conv_in.weight[:, 4:], mode='fan_out')
    unet.conv_in.weight[:, 4:] *= 0.1  # Scale down
unet = unet.to(device)

# %%
# Get one sample
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
unet_input = torch.cat([noisy_latents, gb_latents], dim=1)  # Should be [1, 8, 14, 16]
print(unet_input.shape)

# %%
# Test UNet forward pass
timestep = torch.tensor([100], device=device)
with torch.no_grad():
    # empty text prompt
    text_emb = text_encoder(tokenizer("", return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device))[0]
    
    # Forward pass - this is the critical test
    noise_pred = unet(unet_input, timestep, encoder_hidden_states=text_emb).sample
    
print(f"Input: {unet_input.shape} -> Output: {noise_pred.shape}")

# %% [markdown]
# ## Training Loop

# %%
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import os
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler

# %%
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
    "high quality colorized photograph, natural colors, detailed",
    "photorealistic portrait, natural skin tones, enhanced details",
    "",
    "professional color photography, sharp focus, realistic",
    "detailed color photograph, natural lighting, high resolution",
    "",
]

# %%
def encode_text_batch(prompts):
    """Encode text prompts"""
    text_inputs = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        return text_encoder(text_inputs.input_ids.to(device))[0]

def generate_validation_samples(epoch, num_samples=4):
    """Generate validation samples with proper diffusion inference"""
    print(f"Generating validation samples for epoch {epoch}...")
    unet.eval()
    
    with torch.no_grad():
        val_batch = next(iter(torch.utils.data.DataLoader(dataset_test, batch_size=num_samples)))
        gb_images = val_batch['gb_img'].to(device)
        real_images = val_batch['real_img'].to(device)
        
        gb_latents = vae.encode(gb_images).latent_dist.sample() * scaling_factor
        
        # Start from pure noise (proper diffusion generation)
        sample_latents = torch.randn_like(gb_latents)
        
        # Set up inference scheduler
        inference_scheduler = noise_scheduler
        inference_scheduler.set_timesteps(20)  # 20 inference steps
        
        prompts = [training_prompts[i % len(training_prompts)] for i in range(num_samples)]
        text_embeddings = encode_text_batch(prompts)
        
        # Diffusion denoising loop
        for i, t in enumerate(inference_scheduler.timesteps):
            timestep = t.unsqueeze(0).repeat(num_samples).to(device)
            
            unet_input = torch.cat([sample_latents, gb_latents], dim=1)
            noise_pred = unet(unet_input, timestep, encoder_hidden_states=text_embeddings).sample
            sample_latents = inference_scheduler.step(noise_pred, t, sample_latents).prev_sample
        
        # Decode final result
        generated_images = vae.decode(sample_latents / scaling_factor).sample
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
        
        # Create comparison grid: GB | Real | Generated
        comparison = torch.cat([
            (gb_images * 0.5 + 0.5).clamp(0, 1),
            (real_images * 0.5 + 0.5).clamp(0, 1), 
            generated_images
        ], dim=0)
        
        grid = vutils.make_grid(comparison, nrow=num_samples, padding=2)
        vutils.save_image(grid, os.path.join(config.output_dir, f"validation_epoch_{epoch}.png"))
    
    unet.train()

# %%
print("Starting training...")
global_step = 0
all_losses = []  # Track losses for monitoring

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
        
        # Get text embeddings
        prompts = [training_prompts[i % len(training_prompts)] for i in range(batch_size)]
        text_embeddings = encode_text_batch(prompts)
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
    
    if (epoch + 1) % config.save_image_epochs == 0:
        generate_validation_samples(epoch + 1)

# Final save
unet.save_pretrained(os.path.join(config.output_dir, "final_model"))

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(all_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.savefig(os.path.join(config.output_dir, 'training_loss.png'))
plt.show()

print("Training complete!")
