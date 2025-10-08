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
from degrade_img import GameBoyCameraDegradation
from gameboy_filter import GameBoyCameraConverter

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

def process_file(input_path, cropped_input_dir, output_dir):
    contrast_options = [1.5, 1.8, 2.0]
    brightness_adjust_options = [-10, 0, 10]
    noise_options = [0.01, 0.03, 0.06]
    dither_options = [0.2, 0.25, 0.35, 0.5]
    edge_enhance_options = [False]
    agg_contrast_options = [False, True]
    matrix_size_options = [8, 8, 4]
    rotation_options = [0]

    num_failures = 0
    converter = GameBoyCameraConverter()

    filename_with_extension = os.path.basename(input_path)
    filename, extension = os.path.splitext(filename_with_extension)
    age = int(filename.split("_")[0])
    if age > 70:
        return 0, []


    severities = ['none', 'light', 'medium', 'heavy']
    output_dicts = []
    for severity in severities:
        pil_degrade = degrade_single_image(input_path, severity)
        if pil_degrade is None:
            num_failures += 1
            continue
        contrast = random.choice(contrast_options)
        brightness = random.choice(brightness_adjust_options)
        noise= random.choice(noise_options)
        edge_enhance = random.choice(edge_enhance_options)
        agg_contrast = random.choice(agg_contrast_options)
        dither_strength = random.choice(dither_options)
        matrix_size = random.choice(matrix_size_options)
        rotation = random.choice(rotation_options)

        output_filename = f"{filename}_{severity}{extension}"
        output_path = os.path.join(output_dir, output_filename)
        cropped_input_path = os.path.join(cropped_input_dir, output_filename)


        try:
            cropped, output = converter.convert_image(input_path=input_path, aggressive_contrast=agg_contrast, contrast_boost=contrast, brightness_adjust=brightness, noise_level=noise, edge_enhance=edge_enhance, dithering_strength=dither_strength, matrix_size=matrix_size, rotation_angle=rotation)
            cropped.save(cropped_input_path, quality=100)
            output.save(output_path, quality=100)
        except Exception as e:
            print(e)
            num_failures += 1
            continue

        output_dict = dict(
            contrast=contrast,
            brightness=brightness,
            noise=noise,
            edge_enhance=edge_enhance,
            agg_contrast=agg_contrast,
            severity=severity,
            dither_strength=dither_strength,
            rotation=rotation,
            matrix_size=matrix_size,
            original_input_path=input_path,
            input_path=cropped_input_path,
            output_path=output_path
        )
        output_dicts.append(output_dict)

    return num_failures, output_dicts

def main():
    parser = argparse.ArgumentParser(description='Apply Game Boy camera filter to images')
    parser.add_argument('input_dir', help='Input image folder path')
    parser.add_argument('cropped_input_dir', help='Processed cropped input image folder path')
    parser.add_argument('output_dir', help='Output image folder path')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: input path {args.input_dir} does not exist")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cropped_input_dir, exist_ok=True)

    all_inputs = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    print(f"Found {len(all_inputs)} images in {args.input_dir}")
    print(all_inputs[:3])

    process_func = partial(process_file, output_dir=args.output_dir,  cropped_input_dir=args.cropped_input_dir)
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