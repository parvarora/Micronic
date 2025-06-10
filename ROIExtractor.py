import cv2
class ROIExtractor:
    def __init__(self, template_config):
        self.template_config = template_config
        self.bubble_regions = template_config['bubble_regions']
        
    def extract_all_rois(self, processed_image):
        """Extract all bubble regions from the image"""
        extracted_rois = []
        
        for region in self.bubble_regions:
            roi_data = self._extract_single_roi(processed_image, region)
            extracted_rois.append(roi_data)
        
        return extracted_rois
    
    def _extract_single_roi(self, image, region):
        """Extract a single bubble region with adaptive thresholding"""
        x, y, width, height = region['coordinates']
        
        # Extract ROI with padding
        padding = 5
        roi = image[max(0, y-padding):y+height+padding, 
                   max(0, x-padding):x+width+padding]
        
        # Apply adaptive thresholding
        binary_roi = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Calculate fill percentage
        total_pixels = roi.shape[0] * roi.shape[1]
        dark_pixels = total_pixels - cv2.countNonZero(binary_roi)
        fill_percentage = (dark_pixels / total_pixels) * 100
        
        return {
            'region_id': region['id'],
            'coordinates': region['coordinates'],
            'roi_image': roi,
            'binary_roi': binary_roi,
            'fill_percentage': fill_percentage,
            'question_id': region.get('question_id'),
            'option_value': region.get('option_value')
        }