import cv2
import numpy as np
from skimage import filters, morphology
from scipy import ndimage

class ImageEnhancer:
    def __init__(self):
        self.target_dpi = 300  # Standard DPI for OMR processing
        self.target_contrast = 1.2
        
    def enhance_image(self, image):
        """Complete image enhancement pipeline"""
        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Normalize resolution
        image = self._normalize_resolution(image)
        
        # Step 3: Enhance contrast
        image = self._enhance_contrast(image)
        
        # Step 4: Reduce noise
        image = self._reduce_noise(image)
        
        return image
    
    def _normalize_resolution(self, image):
        """Ensure consistent DPI across all images"""
        height, width = image.shape
        if width < 1200:  # Upscale low-resolution images
            scale_factor = 1200 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        return image
    
    def _enhance_contrast(self, image):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def _reduce_noise(self, image):
        """Multi-stage noise reduction"""
        # Gaussian blur for general noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Morphological operations for salt-and-pepper noise
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image