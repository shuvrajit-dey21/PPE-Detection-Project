"""
Face Recognition Engine
Integrated face recognition system for PPE monitoring with dataset management and training capabilities
"""

import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime
import json


class FaceRecognitionEngine:
    """Face recognition engine with dataset management and training capabilities"""
    
    def __init__(self, dataset_path: str = "face_dataset", model_path: str = "face_model.pkl"):
        """Initialize the face recognition engine
        
        Args:
            dataset_path: Path to store face dataset images
            model_path: Path to save/load the trained model
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            logging.error(f"Failed to load face cascade from {self.cascade_path}")
            raise RuntimeError("Face cascade classifier not found")
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        self.known_faces = {}  # person_id -> name mapping
        self.confidence_threshold = 82  # Confidence threshold for recognition
        
        # Create dataset directory if it doesn't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Load existing model if available
        self.load_model()
    
    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from image

        Args:
            image: Input image as numpy array

        Returns:
            Extracted face image or None if no face found
        """
        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume RGB format (from PIL) and convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # RGBA format, convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                # Single channel, assume it's already grayscale
                gray = image.squeeze()
        else:
            # Already grayscale
            gray = image
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Return the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        return image[y:y+h, x:x+w]
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect all faces in an image and return their locations

        Args:
            image: Input image as numpy array

        Returns:
            List of face detection dictionaries
        """
        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume RGB format (from PIL) and convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # RGBA format, convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                # Single channel, assume it's already grayscale
                gray = image.squeeze()
        else:
            # Already grayscale
            gray = image
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detections = []
        for i, (x, y, w, h) in enumerate(faces):
            face_detections.append({
                'bbox': [x, y, x+w, y+h],
                'confidence': 1.0,  # Haar cascade doesn't provide confidence
                'face_id': i,
                'recognized_person': None,
                'recognition_confidence': 0.0
            })
        
        return face_detections
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect and recognize faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face recognition results
        """
        if not self.is_trained:
            # Return just face detections without recognition
            return self.detect_faces(image)

        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume RGB format (from PIL) and convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # RGBA format, convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                # Single channel, assume it's already grayscale
                gray = image.squeeze()
        else:
            # Already grayscale
            gray = image
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region with padding for better recognition
            padding = int(min(w, h) * 0.1)  # 10% padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)

            face_roi = gray[y_start:y_end, x_start:x_end]

            # Improve face preprocessing for better recognition
            # Apply histogram equalization for better lighting normalization
            face_roi = cv2.equalizeHist(face_roi)

            # Apply Gaussian blur to reduce noise
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)

            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Perform recognition
            person_id, confidence = self.face_recognizer.predict(face_resized)

            # Improved confidence calculation with better scaling
            # Lower confidence values from LBPH mean better matches
            if confidence < 50:
                recognition_confidence = 95  # Very high confidence for very low distance
            elif confidence < 80:
                recognition_confidence = int(90 - (confidence - 50) * 0.5)  # 90-75% range
            elif confidence < 120:
                recognition_confidence = int(75 - (confidence - 80) * 0.75)  # 75-45% range
            elif confidence < 200:
                recognition_confidence = int(45 - (confidence - 120) * 0.3)  # 45-21% range
            else:
                recognition_confidence = max(0, int(20 - (confidence - 200) * 0.1))  # Below 20%

            # Determine if person is recognized with improved threshold logic
            recognized_person = None
            # Use a more lenient threshold for known person IDs
            effective_threshold = self.confidence_threshold
            if person_id in self.known_faces:
                # Reduce threshold by 10% for known person IDs to reduce false negatives
                effective_threshold = max(30, self.confidence_threshold - 10)

            if recognition_confidence > effective_threshold:
                recognized_person = self.known_faces.get(person_id, f"Person_{person_id}")
            
            face_results.append({
                'bbox': [x, y, x+w, y+h],
                'confidence': 1.0,  # Detection confidence
                'face_id': i,
                'recognized_person': recognized_person,
                'recognition_confidence': recognition_confidence,
                'person_id': person_id if recognized_person else None
            })
        
        return face_results
    
    def train_model(self) -> Dict:
        """Train the face recognition model with collected samples
        
        Returns:
            Dictionary with training results
        """
        training_data = []
        labels = []
        person_names = {}
        
        results = {
            'status': 'started',
            'people_trained': 0,
            'total_samples': 0,
            'people_names': []
        }
        
        try:
            # Scan dataset directory for person folders
            person_id = 0
            for person_name in os.listdir(self.dataset_path):
                person_dir = os.path.join(self.dataset_path, person_name)
                
                if not os.path.isdir(person_dir):
                    continue
                
                person_samples = 0
                
                # Load all images for this person
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_dir, filename)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is not None:
                            # Resize to standard size
                            image_resized = cv2.resize(image, (200, 200))
                            training_data.append(image_resized)
                            labels.append(person_id)
                            person_samples += 1
                
                if person_samples > 0:
                    person_names[person_id] = person_name
                    results['people_names'].append(person_name)
                    person_id += 1
                    results['total_samples'] += person_samples
            
            if len(training_data) == 0:
                results['status'] = 'error'
                results['error'] = 'No training data found'
                return results
            
            # Train the model
            labels = np.array(labels, dtype=np.int32)
            self.face_recognizer.train(training_data, labels)
            
            # Save the model and person names
            self.known_faces = person_names
            self.is_trained = True
            self.save_model()
            
            results['status'] = 'completed'
            results['people_trained'] = len(person_names)
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Error during model training: {e}")
        
        return results
    
    def save_model(self) -> bool:
        """Save the trained model and metadata
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if self.is_trained:
                # Save the OpenCV model
                model_xml_path = self.model_path.replace('.pkl', '.xml')
                self.face_recognizer.save(model_xml_path)
                
                # Save metadata
                metadata = {
                    'known_faces': self.known_faces,
                    'confidence_threshold': self.confidence_threshold,
                    'is_trained': self.is_trained,
                    'training_date': datetime.now().isoformat()
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
        
        return False
    
    def load_model(self) -> bool:
        """Load the trained model and metadata

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load metadata
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    metadata = pickle.load(f)

                self.known_faces = metadata.get('known_faces', {})
                self.confidence_threshold = metadata.get('confidence_threshold', 82)
                saved_is_trained = metadata.get('is_trained', False)

                # Load the OpenCV model
                model_xml_path = self.model_path.replace('.pkl', '.xml')
                if os.path.exists(model_xml_path) and saved_is_trained:
                    self.face_recognizer.read(model_xml_path)

                    # Verify that the model is still valid by checking if dataset matches
                    if self._validate_model_with_dataset():
                        self.is_trained = True
                        return True
                    else:
                        # Dataset has changed, model needs retraining
                        self.is_trained = False
                        self.known_faces = {}
                        logging.info("Dataset has changed since last training. Model needs retraining.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

        return False

    def _validate_model_with_dataset(self) -> bool:
        """Validate that the loaded model matches the current dataset

        Returns:
            True if model is valid for current dataset, False otherwise
        """
        try:
            # Get current people in dataset
            current_people = set()
            if os.path.exists(self.dataset_path):
                for person_name in os.listdir(self.dataset_path):
                    person_dir = os.path.join(self.dataset_path, person_name)
                    if os.path.isdir(person_dir):
                        # Check if person has any samples
                        samples = [f for f in os.listdir(person_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if samples:
                            current_people.add(person_name)

            # Get people in trained model
            trained_people = set(self.known_faces.values()) if self.known_faces else set()

            # Model is valid if it contains exactly the same people as the current dataset
            return current_people == trained_people and len(current_people) > 0

        except Exception as e:
            logging.error(f"Error validating model with dataset: {e}")
            return False
    
    def get_dataset_info(self) -> Dict:
        """Get information about the current dataset
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'dataset_path': self.dataset_path,
            'people': [],
            'total_people': 0,
            'total_samples': 0,
            'is_trained': self.is_trained
        }
        
        try:
            for person_name in os.listdir(self.dataset_path):
                person_dir = os.path.join(self.dataset_path, person_name)
                
                if os.path.isdir(person_dir):
                    sample_count = len([f for f in os.listdir(person_dir) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    info['people'].append({
                        'name': person_name,
                        'samples': sample_count
                    })
                    info['total_samples'] += sample_count
            
            info['total_people'] = len(info['people'])
        except Exception as e:
            logging.error(f"Error getting dataset info: {e}")
        
        return info

    def draw_face_detections(self, image: np.ndarray, face_results: List[Dict],
                           show_confidence: bool = True, attendance_info: Dict = None) -> np.ndarray:
        """Draw face detection and recognition results on image

        Args:
            image: Input image as numpy array
            face_results: List of face detection/recognition results
            show_confidence: Whether to show confidence scores
            attendance_info: Optional attendance information for visual indicators

        Returns:
            Image with drawn detections
        """
        result_image = image.copy()

        for face in face_results:
            x1, y1, x2, y2 = face['bbox']

            # Determine color based on recognition status
            if face['recognized_person']:
                color = (0, 255, 0)  # Green for recognized
                label = face['recognized_person']
                if show_confidence:
                    label += f" ({face['recognition_confidence']:.0f}%)"

                # Check if attendance was recently recorded for this person
                attendance_recorded = False
                if attendance_info and 'recent_detections' in attendance_info:
                    for detection in attendance_info['recent_detections']:
                        if detection['name'] == face['recognized_person']:
                            attendance_recorded = True
                            break

                # Enhanced visual indicators for attendance
                if attendance_recorded:
                    # Draw thicker green border for attendance recorded
                    cv2.rectangle(result_image, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 4)
                    # Add "PRESENT" indicator
                    present_label = "✓ PRESENT"
                    present_size = cv2.getTextSize(present_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(result_image, (x2 - present_size[0] - 10, y1),
                                 (x2, y1 + present_size[1] + 10), (0, 255, 0), -1)
                    cv2.putText(result_image, present_label, (x2 - present_size[0] - 5, y1 + present_size[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown Person"

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result_image

    def delete_person_data(self, person_name: str) -> bool:
        """Delete all data for a specific person

        Args:
            person_name: Name of the person to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            person_dir = os.path.join(self.dataset_path, person_name)
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir)

                # Remove from known faces if present
                person_id_to_remove = None
                for person_id, name in self.known_faces.items():
                    if name == person_name:
                        person_id_to_remove = person_id
                        break

                if person_id_to_remove is not None:
                    del self.known_faces[person_id_to_remove]
                    self.save_model()

                return True
        except Exception as e:
            logging.error(f"Error deleting person data: {e}")

        return False

    def update_confidence_threshold(self, threshold: int) -> None:
        """Update the confidence threshold for recognition

        Args:
            threshold: New confidence threshold (0-100)
        """
        self.confidence_threshold = max(0, min(100, threshold))
        if self.is_trained:
            self.save_model()

    def get_model_stats(self) -> Dict:
        """Get statistics about face recognition model performance

        Returns:
            Dictionary with model statistics
        """
        if not self.is_trained:
            return {
                'trained': False,
                'people_count': 0,
                'total_samples': 0,
                'confidence_threshold': self.confidence_threshold
            }

        dataset_info = self.get_dataset_info()
        return {
            'trained': True,
            'people_count': dataset_info['total_people'],
            'total_samples': dataset_info['total_samples'],
            'confidence_threshold': self.confidence_threshold,
            'people_names': [person['name'] for person in dataset_info['people']]
        }

    def collect_face_samples(self, person_name: str, num_samples: int = 100) -> Dict:
        """Collect face samples from webcam for a person

        Args:
            person_name: Name of the person to collect samples for
            num_samples: Number of samples to collect

        Returns:
            Dictionary with collection results
        """
        results = {
            'status': 'started',
            'samples_collected': 0,
            'person_name': person_name,
            'error': None
        }

        try:
            # Create person directory
            person_dir = os.path.join(self.dataset_path, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                results['status'] = 'error'
                results['error'] = 'Could not open camera'
                return results

            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            samples_collected = 0
            frame_count = 0

            print(f"Starting face collection for {person_name}")
            print("Look at the camera and move your head slightly for variety")
            print("Press 'q' to stop early")

            while samples_collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Only process every 5th frame to avoid too similar samples
                if frame_count % 5 != 0:
                    continue

                # Detect faces in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

                if len(faces) > 0:
                    # Use the largest face (assuming it's the main subject)
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face

                    # Extract and save face
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))

                    # Save the face sample
                    filename = f"{person_name}_{samples_collected:04d}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    cv2.imwrite(filepath, face_resized)

                    samples_collected += 1

                    # Draw rectangle around face for visual feedback
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {person_name}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # No face detected
                    cv2.putText(frame, "No face detected - look at camera",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the frame
                cv2.imshow(f'Face Collection - {person_name}', frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    results['status'] = 'interrupted'
                    break

            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

            results['samples_collected'] = samples_collected
            if results['status'] != 'interrupted':
                results['status'] = 'completed'

            print(f"Collection finished. Collected {samples_collected} samples for {person_name}")

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Error during face sample collection: {e}")

            # Cleanup on error
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass

        # After collecting samples, check if model needs retraining
        if results['status'] == 'completed':
            # If model was previously trained, it now needs retraining due to new data
            if self.is_trained and not self._validate_model_with_dataset():
                self.is_trained = False
                logging.info(f"Model needs retraining after adding samples for {person_name}")

        return results

    def get_recognition_stats(self, face_results: List[Dict]) -> Dict:
        """Get statistics about face recognition results

        Args:
            face_results: List of face recognition results

        Returns:
            Dictionary with recognition statistics
        """
        stats = {
            'total_faces': len(face_results),
            'recognized_faces': 0,
            'unknown_faces': 0,
            'recognized_people': [],
            'recognition_rate': 0.0
        }

        for face in face_results:
            if face['recognized_person']:
                stats['recognized_faces'] += 1
                if face['recognized_person'] not in stats['recognized_people']:
                    stats['recognized_people'].append(face['recognized_person'])
            else:
                stats['unknown_faces'] += 1

        if stats['total_faces'] > 0:
            stats['recognition_rate'] = (stats['recognized_faces'] / stats['total_faces']) * 100

        return stats

    def reset_training_status(self) -> None:
        """Reset the training status - useful for debugging or forcing retraining"""
        self.is_trained = False
        self.known_faces = {}
        logging.info("Training status reset. Model will need to be retrained.")

    def get_training_status_info(self) -> Dict:
        """Get detailed information about training status

        Returns:
            Dictionary with training status details
        """
        dataset_info = self.get_dataset_info()

        return {
            'is_trained': self.is_trained,
            'model_valid': self._validate_model_with_dataset() if self.is_trained else False,
            'people_in_dataset': dataset_info['total_people'],
            'samples_in_dataset': dataset_info['total_samples'],
            'people_in_model': len(self.known_faces),
            'needs_training': dataset_info['total_people'] > 0 and not self.is_trained,
            'needs_retraining': self.is_trained and not self._validate_model_with_dataset()
        }
