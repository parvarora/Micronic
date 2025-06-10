import numpy as np
import cv2
import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

class AdvancedMarkClassifier:
    def __init__(self):
        self.cnn_model = self._build_cnn_model()
        self.traditional_classifier = self._build_traditional_classifier()
        self.confidence_threshold = 0.7
        
    def _build_cnn_model(self):
        """CNN model for mark classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: empty, filled, ambiguous
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def classify_marks(self, roi_data_list):
        """Classify all extracted ROIs"""
        results = []
        
        for roi_data in roi_data_list:
            # CNN classification
            cnn_prediction = self._cnn_classify(roi_data)
            
            # Traditional feature-based classification
            traditional_prediction = self._traditional_classify(roi_data)
            
            # Ensemble decision
            final_prediction = self._ensemble_decision(cnn_prediction, traditional_prediction)
            
            results.append({
                'region_id': roi_data['region_id'],
                'question_id': roi_data['question_id'],
                'option_value': roi_data['option_value'],
                'prediction': final_prediction['class'],
                'confidence': final_prediction['confidence'],
                'requires_review': final_prediction['confidence'] < self.confidence_threshold
            })
        
        return results
    
    def _cnn_classify(self, roi_data):
        """CNN-based classification"""
        # Preprocess ROI for CNN
        roi = cv2.resize(roi_data['binary_roi'], (50, 50))
        roi = roi.reshape(1, 50, 50, 1) / 255.0
        
        # Predict
        predictions = self.cnn_model.predict(roi)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        classes = ['empty', 'filled', 'ambiguous']
        return {'class': classes[class_idx], 'confidence': confidence}
    
    def _traditional_classify(self, roi_data):
        """Feature-based classification using traditional CV"""
        fill_percentage = roi_data['fill_percentage']
        
        # Extract additional features
        roi = roi_data['binary_roi']
        
        # Contour analysis
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'fill_percentage': fill_percentage,
            'contour_count': len(contours),
            'largest_contour_area': max([cv2.contourArea(c) for c in contours]) if contours else 0,
            'edge_density': self._calculate_edge_density(roi)
        }
        
        # Rule-based classification
        if fill_percentage > 40:
            return {'class': 'filled', 'confidence': 0.8}
        elif fill_percentage < 15:
            return {'class': 'empty', 'confidence': 0.8}
        else:
            return {'class': 'ambiguous', 'confidence': 0.5}
    
    def _ensemble_decision(self, cnn_pred, traditional_pred):
        """Combine predictions from both classifiers"""
        # Weighted ensemble (CNN gets higher weight)
        cnn_weight = 0.7
        traditional_weight = 0.3
        
        if cnn_pred['class'] == traditional_pred['class']:
            # Both agree - high confidence
            combined_confidence = (cnn_pred['confidence'] * cnn_weight + 
                                 traditional_pred['confidence'] * traditional_weight)
            return {'class': cnn_pred['class'], 'confidence': combined_confidence}
        else:
            # Disagreement - use higher confidence prediction but reduce confidence
            if cnn_pred['confidence'] > traditional_pred['confidence']:
                return {'class': cnn_pred['class'], 
                       'confidence': cnn_pred['confidence'] * 0.7}
            else:
                return {'class': traditional_pred['class'], 
                       'confidence': traditional_pred['confidence'] * 0.7}