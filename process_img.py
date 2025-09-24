#!/usr/bin/env python3
"""
Simplified Game Boy Camera Style Image Filter
Converts images to Game Boy camera aesthetic with grayscale palette.
"""

import cv2
import argparse
import numpy as np
from PIL import Image
import sys
import os
import glob
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from test_degrade import GameBoyCameraDegradation

# Fixed grayscale palette (dark to light)
PALETTE = [(40, 40, 40), (104, 104, 104), (168, 168, 168), (252, 252, 252)]

# 8x8 Bayer dithering matrix
BAYER_MATRIX = np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21]
])

def clamp(value, min_val=0, max_val=255):
    """Clamp value between min_val and max_val"""
    return max(min_val, min(max_val, value))

def apply_levels(value, contrast, gamma):
    """Apply contrast and gamma correction"""
    new_value = value / 255.0
    new_value = (new_value - 0.5) * contrast + 0.5
    new_value = clamp(new_value, 0, 1)
    return int(pow(new_value, gamma) * 255)

def to_grayscale(image, contrast=1.5, gamma=1.0):
    """Convert image to grayscale with contrast and gamma correction"""
    # Convert to grayscale using standard luminance formula
    gray = np.dot(image[...,:3], [0.3, 0.59, 0.11])
    
    # Apply levels correction
    gray_corrected = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gray_corrected[i, j] = apply_levels(gray[i, j], contrast, gamma)
    
    return gray_corrected.astype(np.uint8)

def sharpen_filter(image, sharpness=1.0):
    """Apply sharpening filter"""
    if sharpness == 0:
        return image
    
    height, width = image.shape
    # Convert to float to avoid overflow
    image_float = image.astype(np.float32)
    sharpened = np.zeros_like(image_float)
    
    # Apply sharpening kernel
    for i in range(height):
        for j in range(width):
            # Get neighboring pixels (with bounds checking)
            n = image_float[max(0, i-1), j] - 128  # north
            s = image_float[min(height-1, i+1), j] - 128  # south
            w = image_float[i, max(0, j-1)] - 128  # west
            e = image_float[i, min(width-1, j+1)] - 128  # east
            center = image_float[i, j] - 128
            
            # Apply sharpening formula
            result = center + ((4*center - w - e - n - s) * sharpness)
            sharpened[i, j] = clamp(result, -128, 127) + 128
    
    return sharpened.astype(np.uint8)

def bayer_dither(image, dither_strength=0.6):
    """Apply Bayer matrix dithering"""
    height, width = image.shape
    dithered = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            # Get Bayer matrix value
            bayer_val = BAYER_MATRIX[i % 8, j % 8]
            pixel_val = float(image[i, j])
            
            # Apply dithering
            dithered_val = pixel_val + ((bayer_val - 32) * dither_strength)
            dithered_val = clamp(dithered_val, 0, 255)
            
            # Quantize to 4 levels (0, 85, 170, 255)
            level = clamp(round(dithered_val / 85), 0, 3)
            dithered[i, j] = level * 85
    
    return dithered.astype(np.uint8)

def apply_palette(image):
    """Apply grayscale palette to quantized image"""
    colored = np.zeros((*image.shape, 3), dtype=np.uint8)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Map quantized values to palette indices
            if image[i, j] <= 42:  # ~0-85/2
                palette_index = 0
            elif image[i, j] <= 127:  # ~85/2-170/2  
                palette_index = 1
            elif image[i, j] <= 212:  # ~170/2-255/2
                palette_index = 2
            else:
                palette_index = 3
            
            colored[i, j] = PALETTE[palette_index]
    
    return colored

def resize_to_gameboy(image, target_width=128, target_height=112, up_scale=6):
    """Resize image to Game Boy camera resolution, cropping to fill the frame"""
    pil_image = Image.fromarray(image)

    # Calculate scaling to fill the target dimensions (crop instead of fit)
    width_ratio = target_width / pil_image.width
    height_ratio = target_height / pil_image.height
    scale = max(width_ratio, height_ratio)  # Use max instead of min to fill
    
    new_width = int(pil_image.width * scale)
    new_height = int(pil_image.height * scale)

    # Resize with nearest neighbor for pixelated effect
    resized = pil_image.resize((new_width, new_height), Image.NEAREST)
    
    # Crop to exact target size (center crop)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped = resized.crop((left, top, right, bottom))

    # Get the upscaled cropped image
    target_width_up = int(new_width * up_scale)
    target_height_up = int(new_height * up_scale)
    upscaled = pil_image.resize((target_width_up, target_height_up), Image.NEAREST)

    new_width = int(upscaled.width)
    new_height = int(upscaled.height)

    target_width_up = int(target_width * up_scale)
    target_height_up = int(target_height * up_scale)
    left = (new_width - target_width_up) // 2
    top = (new_height - target_height_up) // 2
    right = left + target_width_up
    bottom = top + target_height_up
    upscaled_crop = upscaled.crop((left, top, right, bottom))

    cropped = np.array(cropped)
    upscaled_crop = np.array(upscaled_crop)
    return cropped, upscaled_crop

def gameboy_camera_filter(input_path, pil_image, altered_input_path, cropped_input_path, output_path, contrast=1.5, gamma=1.0, 
                         sharpness=1.0, dither_strength=0.6, scale_factor=6):
    """
    Apply Game Boy camera filter to an image
    
    Args:
        pil_image: image data
        altered_input_path: Path to the resized image
        output_path: Path to save processed image
        contrast: Contrast adjustment (0.6-2.4)
        gamma: Gamma correction (0.4-2.5)  
        sharpness: Sharpening strength (0-2.0)
        dither_strength: Dithering strength (0-1.0)
        scale_factor: Output scaling factor
    """
    
    try:
        # Load image
        original_image = Image.open(input_path)
        original_image_array = np.array(original_image.convert('RGB')) 
        image_array = np.array(pil_image.convert('RGB'))
        
        # Resize to Game Boy resolution (128x112)
        resized, upscale_crop = resize_to_gameboy(image_array, 128, 112, up_scale=scale_factor)
        _, upscale_crop_og = resize_to_gameboy(original_image_array, 128, 112, up_scale=scale_factor)
        
        # Convert to grayscale with levels adjustment
        gray = to_grayscale(resized, contrast, gamma)
        
        # Apply sharpening
        if sharpness > 0:
            gray = sharpen_filter(gray, sharpness)
        
        # Apply Bayer dithering
        dithered = bayer_dither(gray, dither_strength)
        
        # Apply grayscale palette
        colored = apply_palette(dithered)
        
        # Scale up output and resized input
        output_image = Image.fromarray(colored)
        upscale_crop_img = Image.fromarray(upscale_crop)
        upscale_crop_img_og = Image.fromarray(upscale_crop_og)
        if scale_factor > 1:
            new_size = (output_image.width * scale_factor, output_image.height * scale_factor)
            output_image = output_image.resize(new_size, Image.NEAREST)
        
        # Save result
        upscale_crop_img.save(altered_input_path)
        upscale_crop_img_og.save(cropped_input_path)
        output_image.save(output_path)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False
    
    return True

def degrade_single_image(img_file, severity):
    try:
        # Load and preprocess
        img = cv2.imread(str(img_file))
        if img is None:
            return False
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        degrader = GameBoyCameraDegradation()

        degraded = degrader.degrade_image(
            img_rgb, 
            severity=severity, 
            fast_compression=False
        )

        degraded_uint8 = (degraded * 255).astype(np.uint8)
        pil_image = Image.fromarray(degraded_uint8)
        return pil_image
        
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        return None

def process_file(input_path, altered_input_dir, cropped_input_dir, output_dir):
    contrast_options =  [1.5, 1.6, 1.7] #[1.6, 1.7, 1.8, 1.9, 2.0]
    gamma_options = [0.9, 1.0]  #[0.7, 0.8, 0.9]
    sharpness_options = [0.5, 0.6, 0.7, 0.8] #[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    dither_options = [0.5, 0.6] #[0.4, 0.5, 0.6, 0.7]
    num_failures = 0

    filename_with_extension = os.path.basename(input_path)
    filename, extension = os.path.splitext(filename_with_extension)
    severities = ['none', 'light', 'medium', 'heavy']
    output_dicts = []
    for severity in severities:
        pil_degrade = degrade_single_image(input_path, severity)
        if pil_degrade is None:
            num_failures += 1
            continue
        contrast = random.choice(contrast_options)
        gamma = random.choice(gamma_options)
        sharpness = random.choice(sharpness_options)
        dither = random.choice(dither_options)

        output_filename = f"{filename}_c{contrast}_g{gamma}_s{sharpness}_d{dither}_{severity}{extension}"
        output_path = os.path.join(output_dir, output_filename)
        altered_input_path = os.path.join(altered_input_dir, output_filename)
        cropped_input_path = os.path.join(cropped_input_dir, output_filename)

        success = gameboy_camera_filter(input_path,
            pil_degrade, altered_input_path, cropped_input_path, output_path, contrast, gamma, 
            sharpness, dither, 6
        )
        if not success:
            num_failures += 1

        output_dict = dict(
            contrast=contrast,
            gamma=gamma,
            sharpness=sharpness,
            dither=dither,
            severity=severity,
            original_input_path=input_path,
            degraded_input_path = altered_input_path,
            input_path=cropped_input_path,
            output_path=output_path
        )
        output_dicts.append(output_dict)

    return num_failures, output_dicts

def main():
    parser = argparse.ArgumentParser(description='Apply Game Boy camera filter to images')
    parser.add_argument('input_dir', help='Input image folder path')
    parser.add_argument('altered_input_dir', help='Processed cropped degraded input image folder path')
    parser.add_argument('cropped_input_dir', help='Processed cropped input image folder path')
    parser.add_argument('output_dir', help='Output image folder path')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: input path {args.input_dir} does not exist")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.altered_input_dir, exist_ok=True)
    os.makedirs(args.cropped_input_dir, exist_ok=True)

    all_inputs = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    print(f"Found {len(all_inputs)} images in {args.input_dir}")
    print(all_inputs[:3])


    process_func = partial(process_file, output_dir=args.output_dir, altered_input_dir=args.altered_input_dir, cropped_input_dir=args.cropped_input_dir)
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process_func, all_inputs), 
                        total=len(all_inputs), 
                        desc="Processing images"))
        
    total_failures = sum([r[0] for r in results])
    all_data = [r[1] for r in results]
    all_data_flat = [item for sublist in all_data for item in sublist]
    df = pd.DataFrame(all_data_flat)
    df.to_csv("face_data.csv", index=False)

    print(f"Completed with {total_failures} failures")

if __name__ == '__main__':
    main()