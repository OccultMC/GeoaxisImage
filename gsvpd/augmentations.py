"""
Image augmentation utilities.
Handles pixel-level modifications like noise, blur, color shifts, and compression.
"""
import numpy as np
import cv2
import random

def apply_pixel_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Apply a chain of random augmentations to an image based on specific probabilities.
    
    Probabilities:
    - Blur: 70% None, 15% Tiny, 15% Moderate.
    - Noise: 70% None, 15% Tiny, 15% Moderate.
    - JPEG: 70% None, 30% Random Quality (80-100).
    - Brightness/Contrast: Always applied with random range.
    - Saturation: Always applied with random range.
    
    Args:
        image: Input image (numpy array, BGR)
        
    Returns:
        Augmented image (numpy array, BGR)
    """
    if image is None:
        return image

    output = image.copy()
    
    # 1. Gaussian Blur
    # 70% No blur, 15% Tiny (3px), 15% Bit (5px)
    blur_roll = random.random()
    blur_k = 1
    
    if blur_roll < 0.70:
        blur_k = 1 # No blur
    elif blur_roll < 0.85:
        blur_k = 1 # Tiny blur
    else:
        blur_k = 3 # Bit of blur

    if blur_k > 1:
        output = cv2.GaussianBlur(output, (blur_k, blur_k), 0)
        
    # 2. Add Gaussian Noise
    # 70% No noise, 15% Tiny noise, 15% Bit of noise
    noise_roll = random.random()
    apply_noise = False
    var_min, var_max = 0, 0
    
    if noise_roll < 0.70:
        apply_noise = False
    elif noise_roll < 0.85:
        apply_noise = True
        var_min, var_max = 1, 5 # Tiny noise variance
    else:
        apply_noise = True
        var_min, var_max = 5, 15 # Bit of noise variance

    if apply_noise:
        row, col, ch = output.shape
        mean = 0
        var = random.uniform(var_min, var_max)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        
        # Add noise and clip to valid byte range
        noisy = output.astype(np.float32) + gauss
        output = np.clip(noisy, 0, 255).astype(np.uint8)

    # 3. Brightness and Contrast (Kept original logic)
    # alpha (contrast) [1.0-0.2, 1.0+0.2]
    # beta (brightness) [-20, 20]
    alpha = random.uniform(0.85, 1.15) 
    beta = random.uniform(-15, 15)
    output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    # 4. Saturation Adjustment
    # Random multiplier between 0.8 and 1.2
    sat_mult = random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32) * sat_mult
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 5. JPEG Compression Quality
    # 30% of the time apply compression between 80 and 100
    # 70% of the time do nothing (keep original quality)
    if random.random() > 0.70:
        quality = random.randint(80, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', output, encode_param)
        output = cv2.imdecode(encimg, 1)
        
    return output