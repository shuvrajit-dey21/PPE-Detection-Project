"""
PPE Detection Engine
Core detection engine for PPE compliance monitoring using YOLOv8 with integrated face recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional
import os
import time
import logging

# Import face recognition engine
try:
    from face_recognition_engine import FaceRecognitionEngine
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("Face recognition engine not available")

# Detection Engine Constants
PROGRESS_UPDATE_INTERVAL = 0.5  # seconds between progress updates
DEFAULT_DETECTION_PERSISTENCE = 3  # frames to keep detections visible
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45


class PPEDetectionEngine:
    """Core PPE detection engine using YOLOv8 with integrated face recognition"""

    def __init__(self, model_path: str = "best.pt", enable_face_recognition: bool = True):
        """Initialize the PPE detection engine

        Args:
            model_path: Path to the YOLOv8 model file
            enable_face_recognition: Whether to enable face recognition features
        """
        self.model_path = model_path
        self.model = None
        self.class_names = {
            0: 'Hardhat',
            1: 'Mask',
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Cone',
            7: 'Safety Vest',
            8: 'machinery',
            9: 'vehicle'
        }
        self.compliance_classes = {
            'Hardhat': True,
            'Mask': True,
            'Safety Vest': True,
            'NO-Hardhat': False,
            'NO-Mask': False,
            'NO-Safety Vest': False
        }

        # Initialize face recognition engine
        self.face_recognition_enabled = enable_face_recognition and FACE_RECOGNITION_AVAILABLE
        self.face_engine = None

        if self.face_recognition_enabled:
            try:
                self.face_engine = FaceRecognitionEngine()
                logging.info("Face recognition engine initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize face recognition engine: {e}")
                self.face_recognition_enabled = False

        self.load_model()
    
    def load_model(self) -> bool:
        """Load the YOLOv8 model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                return True
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.5,
                      iou_threshold: float = 0.45) -> Dict:
        """Detect objects in an image

        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS

        Returns:
            Dict containing detection results and compliance info
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Ensure image has correct number of channels (RGB = 3 channels)
            if len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to RGB by removing alpha channel
                image = image[:, :, :3]
            elif len(image.shape) == 2:
                # Convert grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Convert single channel to RGB
                image = np.repeat(image, 3, axis=2)

            # Run inference
            results = self.model(image, conf=conf_threshold, iou=iou_threshold)

            # Process results
            detections = []
            people_detections = []
            violation_detections = []

            compliance_stats = {
                'total_people': 0,
                'compliant_people': 0,
                'violations': [],
                'compliance_rate': 100.0,
                'people_with_violations': 0
            }

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    # First pass: collect all detections and separate people from violations
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        class_name = self.class_names.get(class_id, f"Class_{class_id}")

                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'is_compliant': self.compliance_classes.get(class_name, None)
                        }
                        detections.append(detection)

                        # Separate people and violations for proper association
                        if class_name == 'Person':
                            people_detections.append(detection)
                        elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                            violation_detections.append(detection)

                    # Count total people
                    compliance_stats['total_people'] = len(people_detections)

                    # Associate violations with people using spatial proximity
                    people_with_violations = set()

                    for violation in violation_detections:
                        violation_bbox = violation['bbox']
                        violation_center = [
                            (violation_bbox[0] + violation_bbox[2]) / 2,
                            (violation_bbox[1] + violation_bbox[3]) / 2
                        ]

                        # Find closest person to this violation
                        min_distance = float('inf')
                        closest_person_idx = -1

                        for i, person in enumerate(people_detections):
                            person_bbox = person['bbox']
                            person_center = [
                                (person_bbox[0] + person_bbox[2]) / 2,
                                (person_bbox[1] + person_bbox[3]) / 2
                            ]

                            # Calculate distance between centers
                            distance = ((violation_center[0] - person_center[0]) ** 2 +
                                      (violation_center[1] - person_center[1]) ** 2) ** 0.5

                            # Check if violation is within reasonable proximity to person
                            person_width = person_bbox[2] - person_bbox[0]
                            person_height = person_bbox[3] - person_bbox[1]
                            max_distance = max(person_width, person_height) * 1.5  # 1.5x person size

                            if distance < min_distance and distance < max_distance:
                                min_distance = distance
                                closest_person_idx = i

                        # If we found a person close enough, associate the violation
                        if closest_person_idx >= 0:
                            people_with_violations.add(closest_person_idx)
                            compliance_stats['violations'].append({
                                'type': violation['class_name'],
                                'bbox': violation['bbox'],
                                'confidence': violation['confidence'],
                                'associated_person_idx': closest_person_idx
                            })
                        else:
                            # Violation without associated person (still count it)
                            compliance_stats['violations'].append({
                                'type': violation['class_name'],
                                'bbox': violation['bbox'],
                                'confidence': violation['confidence'],
                                'associated_person_idx': -1
                            })

                    # Calculate accurate compliance statistics
                    compliance_stats['people_with_violations'] = len(people_with_violations)

                    if compliance_stats['total_people'] > 0:
                        compliance_stats['compliant_people'] = compliance_stats['total_people'] - compliance_stats['people_with_violations']
                        compliance_stats['compliance_rate'] = (compliance_stats['compliant_people'] / compliance_stats['total_people']) * 100
                    else:
                        compliance_stats['compliant_people'] = 0
                        compliance_stats['compliance_rate'] = 100.0  # No people = 100% compliance

            # Perform face recognition if enabled
            face_results = []
            face_stats = {
                'total_faces': 0,
                'recognized_faces': 0,
                'unknown_faces': 0,
                'recognized_people': [],
                'recognition_rate': 0.0
            }

            if self.face_recognition_enabled and self.face_engine:
                try:
                    face_results = self.face_engine.recognize_faces(image)
                    face_stats = self.face_engine.get_recognition_stats(face_results)
                except Exception as e:
                    logging.error(f"Face recognition error: {e}")

            return {
                'detections': detections,
                'compliance_stats': compliance_stats,
                'face_results': face_results,
                'face_stats': face_stats,
                'image_shape': image.shape
            }
            
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            if hasattr(image, 'shape'):
                error_msg += f" (Image shape: {image.shape})"
            logging.error(error_msg)
            return {"error": error_msg}
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict],
                       show_confidence: bool = True, face_results: Optional[List[Dict]] = None,
                       attendance_info: Optional[Dict] = None) -> np.ndarray:
        """Draw detection results on image including face recognition

        Args:
            image: Input image
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            face_results: Optional list of face recognition results

        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        # Define colors for different classes
        colors = {
            'Hardhat': (0, 255, 0),      # Green
            'Mask': (0, 255, 0),         # Green
            'Safety Vest': (0, 255, 0),  # Green
            'NO-Hardhat': (0, 0, 255),   # Red
            'NO-Mask': (0, 0, 255),      # Red
            'NO-Safety Vest': (0, 0, 255), # Red
            'Person': (255, 255, 0),     # Yellow
            'Safety Cone': (255, 165, 0), # Orange
            'machinery': (128, 0, 128),   # Purple
            'vehicle': (255, 192, 203)    # Pink
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color
            color = colors.get(class_name, (128, 128, 128))  # Default gray
            
            # Draw bounding box with improved visibility
            x1, y1, x2, y2 = map(int, bbox)

            # Draw thicker, more visible bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

            # Add a subtle shadow/outline for better visibility
            cv2.rectangle(result_image, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 1)

            # Prepare label
            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"

            # Draw label background with better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_size[1] - 12
            label_bg_x2 = x1 + label_size[0] + 8
            label_bg_y2 = y1

            # Draw label background with shadow
            cv2.rectangle(result_image, (label_bg_x1+1, label_bg_y1+1),
                         (label_bg_x2+1, label_bg_y2+1), (0, 0, 0), -1)
            cv2.rectangle(result_image, (label_bg_x1, label_bg_y1),
                         (label_bg_x2, label_bg_y2), color, -1)

            # Draw label text with better visibility
            cv2.putText(result_image, label, (x1 + 4, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw face recognition results if available
        if face_results and self.face_recognition_enabled and self.face_engine:
            result_image = self.face_engine.draw_face_detections(
                result_image, face_results, show_confidence, attendance_info
            )

        return result_image

    def get_face_engine(self) -> Optional[FaceRecognitionEngine]:
        """Get the face recognition engine instance

        Returns:
            Face recognition engine instance or None if not available
        """
        return self.face_engine if self.face_recognition_enabled else None

    def is_face_recognition_enabled(self) -> bool:
        """Check if face recognition is enabled and available

        Returns:
            True if face recognition is enabled and working, False otherwise
        """
        return self.face_recognition_enabled and self.face_engine is not None

    def process_video(self, video_path: str, output_path: str,
                     conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                     progress_callback=None, skip_frames: int = 1,
                     max_resolution: int = 1280, stop_flag=None,
                     persistent_overlay: bool = True) -> Dict:
        """Process video file for PPE detection with optimizations and cancellation

        Args:
            video_path: Path to input video
            output_path: Path for output video
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            progress_callback: Callback function for progress updates
            skip_frames: Process every Nth frame (1 = all frames)
            max_resolution: Maximum resolution for processing (width)
            stop_flag: Threading event to stop processing

        Returns:
            Dict containing processing results and statistics
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Could not open video file"}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate optimal resolution for processing
        if orig_width > max_resolution:
            scale_factor = max_resolution / orig_width
            process_width = max_resolution
            process_height = int(orig_height * scale_factor)
        else:
            scale_factor = 1.0
            process_width = orig_width
            process_height = orig_height

        # Setup video writer with original resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

        # Initialize statistics
        video_stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_violations': 0,
            'frame_violations': [],
            'compliance_timeline': [],
            'processing_fps': 0
        }

        frame_count = 0
        processed_count = 0
        start_time = time.time()
        last_progress_update = start_time

        # For persistent overlay - store last detection results
        last_detections = []
        last_detection_frame = -1
        # Keep detections visible for multiple frames to prevent flickering
        detection_persistence = skip_frames * DEFAULT_DETECTION_PERSISTENCE if persistent_overlay else 0

        try:
            while True:
                # Check for stop signal
                if stop_flag and stop_flag.is_set():
                    video_stats['cancelled'] = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Determine which detections to use for this frame
                current_detections = []

                # Process every skip_frames frame for consistent speed
                should_process = (frame_count % skip_frames == 0)

                if should_process:
                    # Resize frame for processing if needed
                    if scale_factor != 1.0:
                        process_frame = cv2.resize(frame, (process_width, process_height))
                    else:
                        process_frame = frame

                    # Detect objects in frame
                    results = self.detect_objects(process_frame, conf_threshold, iou_threshold)

                    if 'error' not in results:
                        # Scale detection results back to original resolution
                        if scale_factor != 1.0:
                            scaled_detections = []
                            for det in results['detections']:
                                scaled_det = det.copy()
                                bbox = det['bbox']
                                scaled_det['bbox'] = [
                                    bbox[0] / scale_factor,
                                    bbox[1] / scale_factor,
                                    bbox[2] / scale_factor,
                                    bbox[3] / scale_factor
                                ]
                                scaled_detections.append(scaled_det)
                            results['detections'] = scaled_detections

                        # Store detections for persistence
                        last_detections = results['detections'].copy()
                        last_detection_frame = frame_count
                        current_detections = results['detections']

                        # Update statistics with improved accuracy
                        compliance_stats = results['compliance_stats']
                        # Count people with violations instead of total violations for better accuracy
                        video_stats['total_violations'] += compliance_stats['people_with_violations']
                        video_stats['compliance_timeline'].append(compliance_stats['compliance_rate'])

                        if compliance_stats['people_with_violations'] > 0:
                            video_stats['frame_violations'].append({
                                'frame': frame_count,
                                'timestamp': frame_count / fps,
                                'violations': compliance_stats['violations'],
                                'people_with_violations': compliance_stats['people_with_violations']
                            })

                        processed_count += 1
                    else:
                        # Use last detections if available for error frames
                        if persistent_overlay and last_detections and (frame_count - last_detection_frame) <= detection_persistence:
                            current_detections = last_detections
                else:
                    # Skip processing, but use persistent detections if enabled
                    video_stats['skipped_frames'] += 1
                    if persistent_overlay and last_detections and (frame_count - last_detection_frame) <= detection_persistence:
                        current_detections = last_detections

                # Draw detections on frame (either new or persistent)
                if current_detections:
                    processed_frame = self.draw_detections(frame, current_detections)
                else:
                    processed_frame = frame

                # Write frame
                out.write(processed_frame)
                frame_count += 1
                video_stats['processed_frames'] = frame_count

                # Update progress more frequently for smoother display
                current_time = time.time()
                if progress_callback and (current_time - last_progress_update) >= PROGRESS_UPDATE_INTERVAL:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:
                        # Calculate more accurate FPS
                        video_stats['processing_fps'] = frame_count / elapsed_time

                    progress_callback(progress)
                    last_progress_update = current_time

        finally:
            cap.release()
            out.release()

        # Calculate overall statistics
        if video_stats['compliance_timeline']:
            video_stats['average_compliance_rate'] = np.mean(video_stats['compliance_timeline'])
        else:
            video_stats['average_compliance_rate'] = 0.0

        # Add performance metrics
        total_time = time.time() - start_time
        video_stats['processing_time'] = total_time
        video_stats['final_fps'] = frame_count / total_time if total_time > 0 else 0
        video_stats['detection_frames'] = processed_count

        return video_stats
