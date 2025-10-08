#!/usr/bin/env python3
"""
Game Boy Camera Style Image Converter
Transforms images to look like they were taken with a Game Boy Camera
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class GameBoyCameraConverter:
    def __init__(self):
        # Game Boy Camera specifications
        self.gb_width = 128
        self.gb_height = 112
        
        # Authentic 4-level grayscale palette
        # Game Boy Camera used 2-bit depth (4 colors)
        self.palette_authentic = [
            0,    # Black
            84,   # Dark gray
            168,  # Light gray  
            252   # Near white (not pure white for authenticity)
        ]
    
    def create_bayer_matrix(self, size=8):
        """
        Create a Bayer dithering matrix of specified size
        Using 8x8 for more authentic Game Boy Camera look
        """
        if size == 2:
            return np.array([[0, 2], [3, 1]]) / 4
        elif size == 4:
            return np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16
        elif size == 8:
            # 8x8 Bayer matrix for finer dithering
            return np.array([
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ]) / 64
        else:
            raise ValueError(f"Unsupported matrix size: {size}")
    
    def ordered_dither_authentic(self, image, matrix_size=8, aggressive_contrast=False, dithering_strength=0.5):
        """
        Apply ordered (Bayer) dithering with authentic Game Boy Camera characteristics
        """
        img = np.array(image, dtype=float) / 255.0
        h, w = img.shape
        
        # Apply S-curve for more aggressive contrast if enabled
        if aggressive_contrast:
            # Push values toward extremes
            img = np.where(img < 0.5, 
                          img * img * 2,  # Darken shadows more
                          1 - (1 - img) * (1 - img) * 2)  # Brighten highlights more
        
        # Create the Bayer matrix
        bayer_matrix = self.create_bayer_matrix(matrix_size)
        matrix_h, matrix_w = bayer_matrix.shape
        
        # Tile the Bayer matrix to cover the entire image
        tiles_y = (h + matrix_h - 1) // matrix_h
        tiles_x = (w + matrix_w - 1) // matrix_w
        bayer_tiled = np.tile(bayer_matrix, (tiles_y, tiles_x))[:h, :w]
        
        # Add dithering threshold
        threshold_matrix = (bayer_tiled - 0.5) * dithering_strength  # Adjust strength
        dithered = img + threshold_matrix
        
        # Quantize to 4 levels
        quantized = np.zeros_like(dithered)
        thresholds = [0.25, 0.5, 0.75]
        
        quantized[dithered < thresholds[0]] = 0
        quantized[(dithered >= thresholds[0]) & (dithered < thresholds[1])] = 1/3
        quantized[(dithered >= thresholds[1]) & (dithered < thresholds[2])] = 2/3
        quantized[dithered >= thresholds[2]] = 1
        
        # Convert back to palette values
        result = np.zeros_like(img, dtype=np.uint8)
        result[quantized == 0] = self.palette_authentic[0]
        result[quantized == 1/3] = self.palette_authentic[1]
        result[quantized == 2/3] = self.palette_authentic[2]
        result[quantized == 1] = self.palette_authentic[3]
        
        return Image.fromarray(result)
    
    def add_noise_texture(self, image, amount=0.02):
        """
        Add subtle noise to simulate sensor noise from Game Boy Camera
        """
        img_array = np.array(image, dtype=float)
        noise = np.random.normal(0, amount * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(noisy.astype(np.uint8))
    
    def apply_edge_detection(self, image, strength=0.3):
        """
        Apply subtle edge detection to enhance contours
        Game Boy Camera had sharp edge characteristics
        """
        # Find edges
        edges = image.filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(2.0)
        
        # Blend edges with original
        img_array = np.array(image, dtype=float)
        edges_array = np.array(edges, dtype=float)
        
        blended = img_array * (1 - strength) + edges_array * strength
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    
    def convert_image(self, input_path, aggressive_contrast=False,
                    contrast_boost=1.5, brightness_adjust=-10,
                    matrix_size=8, noise_level=0.05, edge_enhance=True, dithering_strength=0.5,
                    scale_factor=6, rotation_angle=0):
        """
        Convert an image to Game Boy Camera style with improved authenticity
        
        Args:
            input_path: Path to input image
            aggressive_contrast: Use aggressive contrast curve
            contrast_boost: Contrast enhancement factor (default: 1.5)
            brightness_adjust: Brightness adjustment (default: -10 for darker look)
            matrix_size: Size of Bayer matrix for ordered dithering (2, 4, or 8)
            noise_level: Amount of noise to add
            edge_enhance: Enhance edges for sharper look
            dithering_strength: Strength of dithering effect
            scale_factor: How much to scale up the final image
            rotation_angle: Rotation angle in degrees (applied before GB filter)
        """
        # Load image
        img = Image.open(input_path)
        
        # Apply rotation FIRST (before any processing)
        if rotation_angle != 0:
            # Use bicubic for smooth rotation
            # Fill with white or middle gray for background
            img = img.rotate(
                rotation_angle, 
                resample=Image.Resampling.BICUBIC,
                fillcolor=(255, 255, 255),  # White fill for rotated corners
                expand=False  # Don't expand canvas (will crop out corners)
            )
        
        current_width = img.width
        current_height = img.height
        target_width = int(self.gb_width * scale_factor)
        target_height = int(self.gb_height * scale_factor)

        width_ratio = target_width / current_width
        height_ratio = target_height / current_height
        scale = max(width_ratio, height_ratio)  # Use max instead of min to fill
        
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        resized = img.resize((new_width, new_height))
        
        # Crop to exact target size (center crop)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        cropped = resized.crop((left, top, right, bottom))
        
        # Store the rotated original for ground truth
        rotated_original = cropped.copy()
        
        # Convert to grayscale
        img = cropped.convert('L')
        
        # Pre-process: Apply slight blur to reduce harsh details
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Resize to Game Boy Camera resolution
        # Use BICUBIC for smoother downsampling
        img = img.resize((self.gb_width, self.gb_height), Image.Resampling.BICUBIC)
        
        # Enhance edges before contrast adjustment if enabled
        if edge_enhance:
            img = self.apply_edge_detection(img, strength=0.2)
        
        # Adjust brightness first (Game Boy Camera images were often dark)
        if brightness_adjust != 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + brightness_adjust / 100)
        
        # Enhance contrast (Game Boy Camera had very high contrast)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_boost)
        
        # Add slight noise before dithering for texture
        img = self.add_noise_texture(img, amount=noise_level)
        
        # Apply dithering
        img = self.ordered_dither_authentic(img, matrix_size=matrix_size, 
                                        aggressive_contrast=aggressive_contrast, 
                                        dithering_strength=dithering_strength)
        
        img = img.convert('L')
        
        # Scale up using nearest neighbor to preserve pixel art aesthetic
        if scale_factor > 1:
            new_size = (self.gb_width * scale_factor, self.gb_height * scale_factor)
            img = img.resize(new_size, Image.Resampling.NEAREST)
        
        # Return both the rotated original (ground truth) and Game Boy version
        return rotated_original, img