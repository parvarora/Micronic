import cv2
import numpy as np
class GeometricCorrector:
    def __init__(self):
        self.max_skew_angle = 15  # Maximum expected skew in degrees
        
    def correct_geometry(self, image, template_data):
        """Complete geometric correction pipeline"""
        # Step 1: Detect and correct skew
        corrected_image, skew_angle = self._correct_skew(image)
        
        # Step 2: Detect registration marks (if available)
        registration_points = self._detect_registration_marks(corrected_image, template_data)
        
        # Step 3: Apply perspective correction
        if registration_points:
            corrected_image = self._correct_perspective(corrected_image, 
                                                     registration_points, 
                                                     template_data)
        
        return corrected_image
    
    def _correct_skew(self, image):
        """Hough line detection for skew correction"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:20]:  # Consider top 20 lines
                angle = theta * 180 / np.pi - 90
                if abs(angle) < self.max_skew_angle:
                    angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            # Rotate image to correct skew
            center = tuple(np.array(image.shape[1::-1]) / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            corrected_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1])
            return corrected_image, median_angle
        
        return image, 0
    
    def _detect_registration_marks(self, image, template_data):
        """Detect corner/registration marks for precise alignment"""
        registration_marks = template_data.get('registration_marks', [])
        detected_points = []
        
        for mark in registration_marks:
            # Template matching for registration marks
            template = mark['template_image']  # Small template of the mark
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.8:  # High confidence threshold
                detected_points.append({
                    'expected': mark['coordinates'],
                    'detected': max_loc,
                    'confidence': max_val
                })
        
        return detected_points
    
    def _correct_perspective(self, image, registration_points, template_data):
        """Apply perspective transformation based on registration marks"""
        if len(registration_points) >= 4:
            # Extract source and destination points
            src_points = np.array([p['detected'] for p in registration_points[:4]], 
                                dtype=np.float32)
            dst_points = np.array([p['expected'] for p in registration_points[:4]], 
                                dtype=np.float32)
            
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            height, width = image.shape
            corrected_image = cv2.warpPerspective(image, matrix, (width, height))
            return corrected_image
        
        return image