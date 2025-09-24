import numpy as np
import cv2
from PIL import Image
import random
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import io

class GameBoyCameraDegradation:
    def __init__(self):
        # Pre-compute reusable kernels and masks
        self.motion_kernels = {}
        self.vignette_masks = {}
        
    def _get_vignette_mask(self, h, w, strength=0.3):
        """Cache vignette masks for reuse"""
        key = (h, w, strength)
        if key not in self.vignette_masks:
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            vignette = 1 - (distance / max_distance) * strength
            self.vignette_masks[key] = np.clip(vignette, 0, 1)
        return self.vignette_masks[key]
    
    def _get_motion_kernel(self, angle, strength):
        """Cache motion blur kernels for reuse"""
        key = (int(angle), int(strength * 10))  # Discretize for caching
        if key not in self.motion_kernels:
            kernel_size = int(strength * 2) + 1
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            
            center = kernel_size // 2
            dx = int(strength * np.cos(np.radians(angle)))
            dy = int(strength * np.sin(np.radians(angle)))
            
            cv2.line(kernel, (center, center), (center + dx, center + dy), 1, 1)
            kernel = kernel / np.sum(kernel)
            self.motion_kernels[key] = kernel
        return self.motion_kernels[key]
        
    def add_sensor_noise(self, image, noise_level=0.15):
        """Optimized noise addition using numpy broadcasting"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        return np.clip(image + noise, 0, 1, out=image)  # In-place clipping
    
    def add_blooming(self, image, threshold=0.8, strength=0.3):
        """Optimized blooming with pre-computed kernels"""
        bright_mask = (image > threshold)
        if len(image.shape) == 3:
            bright_mask = np.any(bright_mask, axis=2, keepdims=False)
        
        # Use fixed kernel size for better performance
        kernel = cv2.getGaussianKernel(5, 5/3, ktype=cv2.CV_32F)
        kernel = kernel @ kernel.T
        
        # Single convolution operation
        bloom_map = cv2.filter2D(bright_mask.astype(np.float32), -1, kernel)
        bloom_map *= strength
        
        if len(image.shape) == 3:
            bloom_map = bloom_map[:, :, np.newaxis]
            
        return np.clip(image + bloom_map, 0, 1)
    
    def add_chromatic_aberration(self, image, strength=2):
        """Optimized chromatic aberration with single transformation matrix"""
        if len(image.shape) != 3:
            return image
            
        h, w = image.shape[:2]
        result = image.copy()
        
        # Only apply to R and B channels (G stays in place)
        for i, offset in enumerate([(-strength, 0), (strength, 0)]):
            if i == 1:  # Skip G channel
                continue
            channel_idx = 0 if i == 0 else 2  # R or B
            dx, dy = offset
            M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            result[:, :, channel_idx] = cv2.warpAffine(
                image[:, :, channel_idx], M, (w, h), 
                borderMode=cv2.BORDER_REFLECT_101
            )
            
        return result
    
    def add_vignetting(self, image, strength=0.3):
        """Use cached vignette mask"""
        h, w = image.shape[:2]
        vignette = self._get_vignette_mask(h, w, strength)
        
        if len(image.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
            
        return image * vignette
    
    def add_motion_blur(self, image, max_blur=3):
        """Optimized motion blur with cached kernels"""
        if random.random() > 0.3:
            return image
            
        angle = random.uniform(0, 360)
        strength = random.uniform(1, max_blur)
        kernel = self._get_motion_kernel(angle, strength)
        
        if len(image.shape) == 3:
            # Process all channels at once
            return cv2.filter2D(image, -1, kernel)
        else:
            return cv2.filter2D(image, -1, kernel)
    
    def add_compression_artifacts_fast(self, image, quality=30):
        """Faster compression simulation using DCT approximation"""
        # Skip actual JPEG compression for speed - approximate with quantization
        if len(image.shape) == 3:
            # Simple quantization that mimics JPEG artifacts
            quantized = np.round(image * (quality / 10)) / (quality / 10)
        else:
            quantized = np.round(image * (quality / 10)) / (quality / 10)
        
        return np.clip(quantized, 0, 1)
    
    def add_compression_artifacts_accurate(self, image, quality=30):
        """More accurate but slower JPEG compression"""
        # Only use when accuracy is critical
        image_uint8 = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            pil_img = Image.fromarray(image_uint8)
        else:
            pil_img = Image.fromarray(image_uint8, mode='L')
            
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        return np.array(compressed, dtype=np.float32) / 255.0
    
    def add_dead_pixels(self, image, num_pixels=None):
        """Vectorized dead pixel addition"""
        if num_pixels is None:
            num_pixels = random.randint(0, 5)
            
        if num_pixels == 0:
            return image
            
        result = image.copy()
        h, w = image.shape[:2]
        
        # Vectorized approach
        pixel_coords = np.random.randint(0, [h, w], size=(num_pixels, 2))
        pixel_values = np.random.rand(num_pixels)
        
        for i in range(num_pixels):
            y, x = pixel_coords[i]
            if pixel_values[i] > 0.5:
                result[y, x] = 0  # Dead pixel
            else:
                if len(image.shape) == 3:
                    result[y, x] = np.random.rand(3)
                else:
                    result[y, x] = 1
                    
        return result
    
    def degrade_image(self, image, severity='medium', fast_compression=True):
        """Optimized degradation pipeline"""
        
        # Ensure correct data type
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        severity_params = {
            'none': {

            },
            'light': {
                'noise': 0.05, 'blooming_strength': 0.1,
                'chroma_strength': 1, 'vignette': 0.1, 'blur': 1, 'compression': 80
            },
            'medium': {
                'noise': 0.10, 'blooming_strength': 0.15,
                'chroma_strength': 2, 'vignette': 0.15, 'blur': 3, 'compression': 60
            },
            'heavy': {
                'noise': 0.15, 'blooming_strength': 0.2,
                'chroma_strength': 3, 'vignette': 0.2, 'blur': 5, 'compression': 40
            }
        }
        
        params = severity_params.get(severity, severity_params['medium'])
        
        # Apply effects in order, modifying image in-place where possible
        degraded = image
        if severity == 'none':
            return degraded
        
        # Lens effects
        degraded = self.add_vignetting(degraded, params['vignette'])
        degraded = self.add_chromatic_aberration(degraded, params['chroma_strength'])
        
        # Motion blur
        degraded = self.add_motion_blur(degraded, params['blur'])
        
        # Sensor effects
        degraded = self.add_sensor_noise(degraded, params['noise'])
        degraded = self.add_blooming(degraded, strength=params['blooming_strength'])
        
        # Compression artifacts
        if fast_compression:
            degraded = self.add_compression_artifacts_fast(degraded, params['compression'])
        else:
            degraded = self.add_compression_artifacts_accurate(degraded, params['compression'])
            
        degraded = self.add_dead_pixels(degraded)
        
        return degraded

def process_single_image(args):
    """Process a single image - designed for multiprocessing"""
    img_file, output_dir, severities, resize_to_gb, fast_compression = args
    
    try:
        # Load and preprocess
        img = cv2.imread(str(img_file))
        if img is None:
            return False
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if resize_to_gb:
            img_rgb = cv2.resize(img_rgb, (112, 128), interpolation=cv2.INTER_AREA)
        
        degrader = GameBoyCameraDegradation()
        base_name = img_file.stem
        
        for idx, severity in enumerate(severities):
            degraded = degrader.degrade_image(
                img_rgb, 
                severity=severity, 
                fast_compression=fast_compression
            )
            
            # Save
            output_file = output_dir / f"{base_name}_degraded_{severity}_{idx}.jpg"
            degraded_uint8 = (degraded * 255).astype(np.uint8)
            degraded_bgr = cv2.cvtColor(degraded_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), degraded_bgr, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])  # High quality save
        
        return True
        
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        return False

def augment_dataset_parallel(input_dir, output_dir, num_processes=None, 
                           fast_compression=True, resize_to_gb=False):
    """Parallel processing version of dataset augmentation"""
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Don't use all cores
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    img_files = []
    for ext in image_extensions:
        img_files.extend(list(input_path.glob(ext)))
    
    if not img_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(img_files)} images")
    print(f"Using {num_processes} processes")
    print(f"Fast compression: {fast_compression}")
    
    # Prepare arguments for multiprocessing
    severities = ['light', 'medium', 'heavy']
    args_list = [
        (img_file, output_path, severities, resize_to_gb, fast_compression) 
        for img_file in img_files
    ]
    
    # Process in parallel
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, args_list),
            total=len(img_files),
            desc="Processing images"
        ))
    
    successful = sum(results)
    print(f"Successfully processed {successful}/{len(img_files)} images")
    print(f"Results saved to {output_dir}")