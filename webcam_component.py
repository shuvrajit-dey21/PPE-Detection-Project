"""
Webcam Component for Real-time PPE Detection
Enhanced webcam functionality with streamlit-webrtc integration
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
from typing import Dict, Any
import time

from ppe_detection_engine import PPEDetectionEngine


def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_analytics_status(frame_count):
    """Get analytics readiness status and requirements"""
    if frame_count == 0:
        return "waiting", "📹 Start camera", "Camera not started"
    elif frame_count < 5:
        return "initializing", "🔄 Initializing", f"Need {5 - frame_count} more frames"
    elif frame_count < 10:
        return "building", "📊 Building report", f"Need {10 - frame_count} more frames for full analytics"
    elif frame_count < 30:
        return "ready", "✅ Analytics ready", "Full analytics available, improving quality"
    else:
        return "optimal", "🎯 High quality", "Optimal analytics with high confidence"


class WebcamPPEDetector:
    """Real-time PPE detection using webcam with streamlit-webrtc"""

    def __init__(self, detection_engine: PPEDetectionEngine, attendance_manager=None):
        self.detection_engine = detection_engine
        self.attendance_manager = attendance_manager
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.show_confidence = True
        self.detection_enabled = True

        # Session tracking
        self.start_time = time.time()
        self.session_id = int(time.time())
        self.last_session_duration = 0.0  # Store the final session duration
        self.session_active = True  # Track if session is currently active

        # Statistics tracking
        self.frame_count = 0
        self.total_people_detected = 0
        self.total_violations_detected = 0
        self.detection_stats = {
            'total_people': 0,
            'total_violations': 0,
            'compliance_rate': 0.0,
            'last_update': time.time(),
            'session_duration': 0.0
        }

        # Face recognition statistics
        self.face_stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'recognition_rate': 0.0,
            'recognized_people': []
        }

        # Attendance tracking
        self.attendance_records = {}  # Track attendance for current session
        self.last_attendance_check = time.time()
        self.attendance_cooldown = 30  # Seconds between attendance records for same person
        self.recent_detections = []  # Store recent face detections for attendance
        self.employee_cache = {}  # Cache employee info to avoid repeated DB lookups
        self.cache_refresh_time = 0  # Last time cache was refreshed
        self.attendance_updated = False  # Flag to trigger UI refresh when attendance is recorded

        # Thread-safe queue for statistics with larger capacity
        self.stats_queue = queue.Queue(maxsize=50)

        # Running averages for smoother display
        self.compliance_history = []
        self.people_history = []
        self.violation_history = []
        self.face_recognition_history = []
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update detection settings"""
        self.conf_threshold = settings.get('conf_threshold', 0.5)
        self.iou_threshold = settings.get('iou_threshold', 0.45)
        self.show_confidence = settings.get('show_confidence', True)
        self.detection_enabled = settings.get('detection_enabled', True)
    
    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frame for PPE detection"""
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Always increment frame count to show activity
        self.frame_count += 1

        if self.detection_enabled and self.detection_engine:
            try:
                # Perform detection
                results = self.detection_engine.detect_objects(
                    img, 
                    self.conf_threshold, 
                    self.iou_threshold
                )
                
                if 'error' not in results:
                    # Draw detections including face recognition
                    face_results = results.get('face_results', [])

                    # Get attendance info for visual indicators (only every 30 frames to avoid blocking)
                    attendance_info = None
                    if self.attendance_manager and self.frame_count % 30 == 0:
                        try:
                            attendance_info = {'recent_detections': self.recent_detections.copy()}
                        except:
                            attendance_info = None

                    img = self.detection_engine.draw_detections(
                        img,
                        results['detections'],
                        self.show_confidence,
                        face_results,
                        attendance_info
                    )
                    
                    # Update statistics
                    compliance_stats = results['compliance_stats']
                    # Frame count already incremented above

                    # Update cumulative statistics correctly
                    # Only count unique people per frame, not accumulate across frames
                    current_people = compliance_stats['total_people']
                    current_violations = compliance_stats['people_with_violations']  # Use people with violations, not total violations

                    # Track maximum people seen in any single frame for session totals
                    if current_people > 0:
                        # Update session totals with current frame data
                        self.total_people_detected = max(self.total_people_detected, current_people)
                        if current_violations > 0:
                            self.total_violations_detected += current_violations

                    # Update face recognition statistics
                    face_stats = results.get('face_stats', {})
                    if face_stats:
                        self.face_stats.update(face_stats)
                        self.face_recognition_history.append(face_stats.get('recognition_rate', 0.0))

                    # Process attendance tracking (more frequent for better responsiveness)
                    if self.frame_count % 5 == 0:  # Process every 5th frame for better responsiveness
                        self._process_attendance_async(face_results)

                    # Update running averages (keep last 30 frames for smoothing)
                    self.compliance_history.append(compliance_stats['compliance_rate'])
                    self.people_history.append(current_people)
                    self.violation_history.append(current_violations)

                    # Keep only recent history
                    max_history = 30
                    if len(self.compliance_history) > max_history:
                        self.compliance_history = self.compliance_history[-max_history:]
                        self.people_history = self.people_history[-max_history:]
                        self.violation_history = self.violation_history[-max_history:]
                        self.face_recognition_history = self.face_recognition_history[-max_history:]

                    # Calculate smoothed averages
                    avg_compliance = sum(self.compliance_history) / len(self.compliance_history)
                    avg_people = sum(self.people_history) / len(self.people_history)
                    avg_violations = sum(self.violation_history) / len(self.violation_history)
                    avg_face_recognition = sum(self.face_recognition_history) / len(self.face_recognition_history) if self.face_recognition_history else 0.0

                    # Update session duration
                    session_duration = time.time() - self.start_time

                    # Create accurate current stats with better calculations
                    current_stats = {
                        'frame_count': self.frame_count,
                        'total_people': compliance_stats['total_people'],  # Current frame people count
                        'violations': compliance_stats['people_with_violations'],  # Current frame people with violations
                        'compliance_rate': compliance_stats['compliance_rate'],  # Current frame compliance
                        'avg_compliance_rate': round(avg_compliance, 1),  # Session average compliance
                        'avg_people': round(avg_people, 1),  # Session average people per frame
                        'avg_violations': round(avg_violations, 1),  # Session average violations per frame
                        'session_duration': round(session_duration, 1),  # Total session time
                        'total_people_session': self.total_people_detected,  # Cumulative people detected
                        'total_violations_session': self.total_violations_detected,  # Cumulative violations
                        'timestamp': time.time(),
                        # Additional accuracy metrics
                        'frames_with_people': sum(1 for p in self.people_history if p > 0),
                        'frames_with_violations': sum(1 for v in self.violation_history if v > 0),
                        'detection_rate': (len([p for p in self.people_history if p > 0]) / len(self.people_history) * 100) if self.people_history else 0,
                        # Face recognition statistics
                        'face_stats': self.face_stats.copy(),
                        'avg_face_recognition': round(avg_face_recognition, 1)
                    }

                    # Add to queue (non-blocking)
                    try:
                        self.stats_queue.put_nowait(current_stats)
                    except queue.Full:
                        # Remove oldest items and add new one
                        try:
                            # Clear multiple old items to prevent queue backup
                            for _ in range(5):
                                self.stats_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.stats_queue.put_nowait(current_stats)
                        except queue.Full:
                            pass
                
            except Exception as e:
                # Draw error message on frame
                cv2.putText(img, f"Detection Error: {str(e)[:50]}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Even when detection is disabled, update basic stats to show activity
            if self.frame_count % 10 == 0:  # Update every 10 frames to avoid spam
                basic_stats = {
                    'frame_count': self.frame_count,
                    'total_people': 0,
                    'violations': 0,
                    'compliance_rate': 100.0,
                    'session_duration': time.time() - self.start_time,
                    'total_people_session': 0,
                    'total_violations_session': 0,
                    'avg_compliance_rate': 100.0
                }

                try:
                    self.stats_queue.put_nowait(basic_stats)
                except queue.Full:
                    pass  # Queue is full, skip this update

        # Add frame info and status overlay
        status_text = "PPE Detection ON" if self.detection_enabled else "PPE Detection OFF"
        status_color = (0, 255, 0) if self.detection_enabled else (0, 165, 255)

        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(img, f"Frame: {self.frame_count}",
                   (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _process_attendance_async(self, face_results):
        """Process attendance asynchronously to avoid blocking video stream"""
        if not self.attendance_manager or not face_results:
            return

        # Quick check for recognized faces with adaptive threshold for better detection
        recognized_faces = [
            face for face in face_results
            if face.get('recognized_person') and face.get('recognition_confidence', 0) > 55  # Further lowered for better responsiveness
        ]

        if recognized_faces:
            # Use threading to avoid blocking the video stream
            import threading
            thread = threading.Thread(target=self._process_attendance, args=(recognized_faces,))
            thread.daemon = True
            thread.start()

    def _process_attendance(self, face_results):
        """Process face recognition results for attendance tracking (optimized)"""
        if not self.attendance_manager or not face_results:
            return

        current_time = time.time()

        # Process each recognized face
        for face in face_results:
            if face.get('recognized_person') and face.get('recognition_confidence', 0) > 55:  # Lowered threshold for better responsiveness
                person_name = face['recognized_person']
                confidence = face['recognition_confidence']

                # Quick cooldown check (reduced cooldown for better responsiveness)
                last_record_time = self.attendance_records.get(person_name, 0)
                if current_time - last_record_time <= max(30, self.attendance_cooldown // 2):  # Reduced cooldown
                    continue  # Skip if still in cooldown

                try:
                    # Get employee information with caching
                    employee_info = self._get_employee_cached(person_name)

                    if employee_info:
                        # Record attendance (this is the potentially slow operation)
                        success = self.attendance_manager.record_attendance(
                            employee_info['employee_id'],
                            confidence,
                            "Main Camera",
                            str(self.session_id)
                        )

                        if success:
                            self.attendance_records[person_name] = current_time

                            # Add to recent detections for UI display
                            detection_info = {
                                'name': person_name,
                                'employee_id': employee_info['employee_id'],
                                'department': employee_info.get('department', 'Not Specified'),
                                'timestamp': current_time,
                                'confidence': confidence,
                                'status': 'Present'
                            }

                            self.recent_detections.append(detection_info)

                            # Keep only last 10 detections
                            if len(self.recent_detections) > 10:
                                self.recent_detections.pop(0)

                            # Log successful attendance recording
                            print(f"✅ Attendance recorded: {person_name} ({confidence:.1f}%) at {time.strftime('%H:%M:%S')}")

                            # Set flag to trigger UI refresh
                            self.attendance_updated = True
                        else:
                            print(f"❌ Failed to record attendance for {person_name}")
                    else:
                        print(f"⚠️ Employee info not found for {person_name}")

                except Exception as e:
                    # Don't let attendance errors crash the video stream
                    import logging
                    logging.error(f"Attendance processing error: {e}")
                    continue

    def _get_employee_cached(self, person_name):
        """Get employee info with caching to improve performance"""
        current_time = time.time()

        # Refresh cache every 5 minutes
        if current_time - self.cache_refresh_time > 300:
            self.employee_cache.clear()
            self.cache_refresh_time = current_time

        # Check cache first
        if person_name in self.employee_cache:
            return self.employee_cache[person_name]

        # Get from database and cache
        employee_info = self.attendance_manager.get_employee_by_name(person_name)
        if employee_info:
            self.employee_cache[person_name] = employee_info

        return employee_info

    def get_attendance_summary(self):
        """Get attendance summary for current session (optimized)"""
        if not self.attendance_manager:
            return {}

        try:
            # Only get fresh data occasionally to avoid blocking
            current_time = time.time()
            if not hasattr(self, '_last_summary_time'):
                self._last_summary_time = 0
                self._cached_summary = {}

            # Refresh summary every 10 seconds
            if current_time - self._last_summary_time > 10:
                today_attendance = self.attendance_manager.get_today_attendance()
                stats = self.attendance_manager.get_attendance_stats()

                self._cached_summary = {
                    'today_attendance': today_attendance,
                    'stats': stats,
                    'recent_detections': self.recent_detections.copy(),
                    'attendance_updated': self.attendance_updated
                }
                self._last_summary_time = current_time

            # Always update recent detections and attendance flag
            self._cached_summary['recent_detections'] = self.recent_detections.copy()
            self._cached_summary['attendance_updated'] = self.attendance_updated
            return self._cached_summary

        except Exception as e:
            import logging
            logging.error(f"Error getting attendance summary: {e}")
            return {'recent_detections': self.recent_detections.copy(), 'attendance_updated': False}

    def check_and_reset_attendance_update(self) -> bool:
        """Check if attendance was updated and reset the flag"""
        if self.attendance_updated:
            self.attendance_updated = False
            return True
        return False

    def get_latest_stats(self) -> Dict[str, Any]:
        """Get latest detection statistics"""
        latest_stats = None

        # Get all available stats (keep only the latest)
        while not self.stats_queue.empty():
            try:
                latest_stats = self.stats_queue.get_nowait()
            except queue.Empty:
                break

        # If no stats in queue but we have frame count, create basic stats
        if latest_stats is None and self.frame_count > 0:
            # Use preserved session duration if session is not active, otherwise calculate current duration
            if self.session_active:
                current_duration = time.time() - self.start_time
            else:
                current_duration = self.last_session_duration

            latest_stats = {
                'frame_count': self.frame_count,
                'total_people': 0,
                'violations': 0,
                'compliance_rate': 100.0,
                'session_duration': current_duration,
                'total_people_session': self.total_people_detected,
                'total_violations_session': self.total_violations_detected,
                'avg_compliance_rate': sum(self.compliance_history) / len(self.compliance_history) if self.compliance_history else 100.0
            }

        return latest_stats

    def stop_session(self):
        """Stop the current session and preserve the final duration"""
        if self.session_active:
            self.last_session_duration = time.time() - self.start_time
            self.session_active = False

    def start_session(self):
        """Start a new session"""
        self.start_time = time.time()
        self.session_active = True
        self.last_session_duration = 0.0

    def reset_stats(self):
        """Reset detection statistics"""
        self.frame_count = 0
        self.total_people_detected = 0
        self.total_violations_detected = 0
        self.start_time = time.time()
        self.session_id = int(time.time())
        self.last_session_duration = 0.0
        self.session_active = True

        # Reset history
        self.compliance_history = []
        self.people_history = []
        self.violation_history = []

        self.detection_stats = {
            'total_people': 0,
            'total_violations': 0,
            'compliance_rate': 0.0,
            'last_update': time.time(),
            'session_duration': 0.0
        }

        # Clear queue
        while not self.stats_queue.empty():
            try:
                self.stats_queue.get_nowait()
            except queue.Empty:
                break


def generate_analysis_summary(stats):
    """Generate intelligent analysis summary from detection statistics"""
    session_stats = stats["session_summary"]
    detection_stats = stats["detection_statistics"]

    # Calculate key metrics
    total_frames = session_stats["total_frames_processed"]
    avg_compliance = detection_stats["average_compliance_rate"]
    total_violations = detection_stats["total_violations_detected"]

    # Generate insights
    insights = []

    # Session quality assessment
    if total_frames > 100:
        insights.append("✅ Comprehensive session with sufficient data for analysis")
    elif total_frames > 30:
        insights.append("⚠️ Moderate session length - consider longer monitoring for better insights")
    else:
        insights.append("⚠️ Short session - extend monitoring time for more reliable analysis")

    # Compliance assessment
    if avg_compliance >= 90:
        insights.append("🟢 Excellent compliance rate - safety protocols are well followed")
    elif avg_compliance >= 75:
        insights.append("🟡 Good compliance rate - minor improvements needed")
    elif avg_compliance >= 50:
        insights.append("🟠 Fair compliance rate - significant safety improvements required")
    else:
        insights.append("🔴 Poor compliance rate - immediate safety intervention needed")

    # Violation analysis
    violation_rate = (total_violations / total_frames * 100) if total_frames > 0 else 0
    if violation_rate == 0:
        insights.append("✅ No safety violations detected during session")
    elif violation_rate < 5:
        insights.append("✅ Very low violation rate - excellent safety performance")
    elif violation_rate < 15:
        insights.append("⚠️ Moderate violation rate - review safety procedures")
    else:
        insights.append("❌ High violation rate - urgent safety training required")

    return {
        "overall_assessment": insights[1] if len(insights) > 1 else "Analysis completed",
        "key_insights": insights,
        "recommendations": generate_recommendations(detection_stats),
        "risk_level": assess_risk_level(avg_compliance, violation_rate)
    }


def generate_recommendations(detection_stats):
    """Generate actionable recommendations based on detection statistics"""
    recommendations = []
    avg_compliance = detection_stats["average_compliance_rate"]

    if avg_compliance < 80:
        recommendations.append("Implement additional PPE training programs")
        recommendations.append("Increase safety awareness campaigns")
        recommendations.append("Consider mandatory safety briefings")

    if detection_stats["total_violations_detected"] > 0:
        recommendations.append("Review and reinforce PPE usage policies")
        recommendations.append("Install additional safety signage")
        recommendations.append("Implement regular safety audits")

    if detection_stats["compliance_variance"] > 400:  # High variance
        recommendations.append("Investigate causes of compliance fluctuations")
        recommendations.append("Standardize safety procedures across shifts")

    if not recommendations:
        recommendations.append("Maintain current excellent safety standards")
        recommendations.append("Continue regular monitoring and assessment")

    return recommendations


def assess_risk_level(compliance_rate, violation_rate):
    """Assess overall risk level based on compliance and violation rates"""
    if compliance_rate >= 90 and violation_rate < 5:
        return {"level": "LOW", "color": "#4CAF50", "description": "Excellent safety performance"}
    elif compliance_rate >= 75 and violation_rate < 15:
        return {"level": "MEDIUM", "color": "#ff9800", "description": "Acceptable with room for improvement"}
    else:
        return {"level": "HIGH", "color": "#f44336", "description": "Requires immediate attention"}


def create_csv_export(stats):
    """Create CSV data for export"""
    import pandas as pd

    # Create timeline data
    timeline_data = []
    compliance_history = stats["historical_data"]["compliance_timeline"]
    people_history = stats["historical_data"]["people_detection_timeline"]
    violation_history = stats["historical_data"]["violation_timeline"]

    max_length = max(len(compliance_history), len(people_history), len(violation_history))

    for i in range(max_length):
        row = {
            "frame_number": i + 1,
            "compliance_rate": compliance_history[i] if i < len(compliance_history) else None,
            "people_detected": people_history[i] if i < len(people_history) else None,
            "violations_detected": violation_history[i] if i < len(violation_history) else None
        }
        timeline_data.append(row)

    df = pd.DataFrame(timeline_data)
    return df.to_csv(index=False)


def create_text_summary(stats):
    """Create human-readable text summary"""
    summary = f"""
PPE DETECTION ANALYSIS REPORT
============================

Export Information:
- Generated: {stats['export_info']['export_timestamp']}
- Session ID: {stats['export_info']['session_id']}

Session Summary:
- Total Frames Processed: {stats['session_summary']['total_frames_processed']:,}
- Session Duration: {stats['session_summary']['session_duration_formatted']}
- Average FPS: {stats['session_summary']['average_fps']:.1f}

Detection Results:
- Total People Detected: {stats['detection_statistics']['total_people_detected']:,}
- Total Violations: {stats['detection_statistics']['total_violations_detected']:,}
- Current Compliance Rate: {stats['detection_statistics']['current_compliance_rate']:.1f}%
- Average Compliance Rate: {stats['detection_statistics']['average_compliance_rate']:.1f}%
- Peak Compliance: {stats['detection_statistics']['peak_compliance_rate']:.1f}%
- Lowest Compliance: {stats['detection_statistics']['lowest_compliance_rate']:.1f}%

Analysis Summary:
- Overall Assessment: {stats['analysis_summary']['overall_assessment']}
- Risk Level: {stats['analysis_summary']['risk_level']['level']} - {stats['analysis_summary']['risk_level']['description']}

Key Insights:
"""

    for insight in stats['analysis_summary']['key_insights']:
        summary += f"- {insight}\n"

    summary += "\nRecommendations:\n"
    for rec in stats['analysis_summary']['recommendations']:
        summary += f"- {rec}\n"

    summary += f"""
Detection Settings:
- Confidence Threshold: {stats['detection_settings']['confidence_threshold']}
- IoU Threshold: {stats['detection_settings']['iou_threshold']}
- Detection Model: {stats['detection_settings']['detection_model']}

Report generated by PPE Monitor Pro
"""

    return summary


def show_export_popup_notification():
    """Show a floating animated popup notification for export"""

    # Set popup state to show
    st.session_state.show_export_popup = True

    # Show the popup modal
    if st.session_state.show_export_popup:
        @st.dialog("🚀 Export Ready!")
        def export_popup():
            # Custom CSS for the popup
            st.markdown("""
            <style>
            /* Popup container styling */
            .stDialog > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border-radius: 20px !important;
                border: none !important;
                box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4) !important;
                animation: popupSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
                backdrop-filter: blur(20px) !important;
            }

            /* Popup content styling */
            .stDialog h1 {
                color: white !important;
                text-align: center !important;
                font-size: 2rem !important;
                margin-bottom: 1.5rem !important;
                text-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            }

            /* Animation keyframes */
            @keyframes popupSlideIn {
                0% {
                    opacity: 0;
                    transform: translateY(-50px) scale(0.9);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }

            @keyframes glow {
                0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
                50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
            }

            /* Success icon styling */
            .popup-icon {
                font-size: 4rem;
                text-align: center;
                margin-bottom: 1rem;
                animation: float 2s ease-in-out infinite;
            }

            /* Message styling */
            .popup-message {
                color: white;
                text-align: center;
                font-size: 1.2rem;
                margin-bottom: 2rem;
                line-height: 1.6;
                text-shadow: 0 1px 5px rgba(0,0,0,0.3);
            }

            /* Button container */
            .popup-buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin-top: 2rem;
            }

            /* Custom button styling */
            .popup-btn {
                padding: 12px 24px;
                border-radius: 12px;
                border: none;
                font-weight: 600;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                text-decoration: none;
                display: inline-block;
                text-align: center;
                min-width: 120px;
            }

            .popup-btn-primary {
                background: rgba(255, 255, 255, 0.9);
                color: #667eea;
                box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
            }

            .popup-btn-primary:hover {
                background: white;
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(255, 255, 255, 0.4);
            }

            .popup-btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }

            .popup-btn-secondary:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            }
            </style>
            """, unsafe_allow_html=True)

            # Popup content
            st.markdown('<div class="popup-icon">🎉</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="popup-message">
                <strong>Your export is ready!</strong><br>
                Click below to access the full export interface with enhanced download options,
                analytics charts, and detailed session reports.
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Go to Export Tab",
                           type="primary",
                           use_container_width=True,
                           key="popup_export_btn"):
                    st.session_state.show_export_popup = False
                    # Set flag to show success message and highlight export tab
                    st.session_state.export_popup_confirmed = True
                    st.rerun()

            with col2:
                if st.button("✕ Cancel",
                           use_container_width=True,
                           key="popup_cancel_btn"):
                    st.session_state.show_export_popup = False
                    st.rerun()

        export_popup()


def create_comprehensive_export_data(latest_stats, webcam_detector):
    """Create comprehensive export data for the new export tab"""
    import numpy as np
    from datetime import datetime
    import time

    # Get fresh stats or create basic stats if needed
    if not latest_stats:
        latest_stats = webcam_detector.get_latest_stats()

    # If still no stats but we have frames, create basic export data
    if not latest_stats and webcam_detector.frame_count > 0:
        latest_stats = {
            'frame_count': webcam_detector.frame_count,
            'total_people': 0,
            'violations': 0,
            'compliance_rate': 100.0,
            'session_duration': time.time() - webcam_detector.start_time,
            'total_people_session': webcam_detector.total_people_detected,
            'total_violations_session': webcam_detector.total_violations_detected,
            'avg_compliance_rate': sum(webcam_detector.compliance_history) / len(webcam_detector.compliance_history) if webcam_detector.compliance_history else 100.0
        }

    # Create comprehensive export data
    export_timestamp = datetime.now()
    session_duration = latest_stats.get('session_duration', 0)

    # Detailed statistics
    detailed_stats = {
        "export_info": {
            "export_timestamp": export_timestamp.isoformat(),
            "export_date": export_timestamp.strftime("%Y-%m-%d"),
            "export_time": export_timestamp.strftime("%H:%M:%S"),
            "session_id": webcam_detector.session_id
        },
        "session_summary": {
            "total_frames_processed": latest_stats['frame_count'],
            "session_duration_seconds": session_duration,
            "session_duration_formatted": format_duration(session_duration),
            "average_fps": latest_stats['frame_count'] / session_duration if session_duration > 0 else 0,
            "detection_start_time": webcam_detector.start_time,
            "detection_end_time": time.time()
        },
        "detection_statistics": {
            "total_people_detected": latest_stats.get('total_people_session', 0),
            "total_violations_detected": latest_stats.get('total_violations_session', 0),
            "current_compliance_rate": latest_stats.get('compliance_rate', 0),
            "average_compliance_rate": latest_stats.get('avg_compliance_rate', 0),
            "peak_compliance_rate": max(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0,
            "lowest_compliance_rate": min(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0,
            "compliance_variance": np.var(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0
        },
        "detection_settings": {
            "confidence_threshold": webcam_detector.conf_threshold,
            "iou_threshold": webcam_detector.iou_threshold,
            "detection_model": "YOLOv8 PPE Detection"
        },
        "historical_data": {
            "compliance_timeline": webcam_detector.compliance_history,
            "people_detection_timeline": webcam_detector.people_history,
            "violation_timeline": webcam_detector.violation_history
        }
    }

    # Create analysis summary
    analysis_summary = generate_analysis_summary(detailed_stats)
    detailed_stats["analysis_summary"] = analysis_summary

    return detailed_stats


def create_comprehensive_export(latest_stats, webcam_detector):
    """Create comprehensive export functionality with downloadable files"""
    import json
    import numpy as np
    import plotly.graph_objects as go
    from datetime import datetime

    # Get fresh stats or create basic stats if needed
    if not latest_stats:
        latest_stats = webcam_detector.get_latest_stats()

    # If still no stats but we have frames, create basic export data
    if not latest_stats and webcam_detector.frame_count > 0:
        latest_stats = {
            'frame_count': webcam_detector.frame_count,
            'total_people': 0,
            'violations': 0,
            'compliance_rate': 100.0,
            'session_duration': time.time() - webcam_detector.start_time,
            'total_people_session': webcam_detector.total_people_detected,
            'total_violations_session': webcam_detector.total_violations_detected,
            'avg_compliance_rate': sum(webcam_detector.compliance_history) / len(webcam_detector.compliance_history) if webcam_detector.compliance_history else 100.0
        }

    if not latest_stats or latest_stats.get('frame_count', 0) == 0:
        st.warning("⚠️ No detection data available to export.")
        st.info("💡 **Tip**: Start the camera and wait for frames to be processed before exporting.")

        # Show basic information if detector has frames but no stats
        if webcam_detector.frame_count > 0:
            st.info(f"🔍 **Status**: {webcam_detector.frame_count} frames processed, but no analytics data available yet.")
            st.info("⏱️ **Solution**: Wait a few more seconds for analytics to initialize.")
        else:
            st.info("📹 **Status**: No frames have been processed yet. Start the camera first.")
        return

    try:
        st.markdown("---")
        st.markdown("### 📊 **Export Detection Statistics**")

        # Create comprehensive export data
        export_timestamp = datetime.now()
        session_duration = latest_stats.get('session_duration', 0)

        # Detailed statistics
        detailed_stats = {
            "export_info": {
                "export_timestamp": export_timestamp.isoformat(),
                "export_date": export_timestamp.strftime("%Y-%m-%d"),
                "export_time": export_timestamp.strftime("%H:%M:%S"),
                "session_id": webcam_detector.session_id
            },
            "session_summary": {
                "total_frames_processed": latest_stats['frame_count'],
                "session_duration_seconds": session_duration,
                "session_duration_formatted": format_duration(session_duration),
                "average_fps": latest_stats['frame_count'] / session_duration if session_duration > 0 else 0,
                "detection_start_time": webcam_detector.start_time,
                "detection_end_time": time.time()
            },
            "detection_statistics": {
                "total_people_detected": latest_stats.get('total_people_session', 0),
                "total_violations_detected": latest_stats.get('total_violations_session', 0),
                "current_compliance_rate": latest_stats.get('compliance_rate', 0),
                "average_compliance_rate": latest_stats.get('avg_compliance_rate', 0),
                "peak_compliance_rate": max(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0,
                "lowest_compliance_rate": min(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0,
                "compliance_variance": np.var(webcam_detector.compliance_history) if webcam_detector.compliance_history else 0
            },
            "detection_settings": {
                "confidence_threshold": webcam_detector.conf_threshold,
                "iou_threshold": webcam_detector.iou_threshold,
                "detection_model": "YOLOv8 PPE Detection"
            },
            "historical_data": {
                "compliance_timeline": webcam_detector.compliance_history,
                "people_detection_timeline": webcam_detector.people_history,
                "violation_timeline": webcam_detector.violation_history
            }
        }

        # Create analysis summary
        analysis_summary = generate_analysis_summary(detailed_stats)
        detailed_stats["analysis_summary"] = analysis_summary

        # Enhanced export options with better information
        st.markdown("#### 📥 **Download Options**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📄 **JSON Report**")
            json_data = json.dumps(detailed_stats, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"ppe_detection_report_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                help="Complete detection data in JSON format for developers and analysts"
            )
            st.caption(f"📊 Size: {len(json_data)/1024:.1f} KB")
            st.caption("🔧 **Contains:** All raw data, settings, and analytics")

        with col2:
            st.markdown("##### 📊 **CSV Data**")
            # Create CSV data with error handling
            try:
                csv_data = create_csv_export(detailed_stats)
                st.download_button(
                    label="📥 Download CSV Data",
                    data=csv_data,
                    file_name=f"ppe_detection_data_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Detection timeline data for spreadsheet analysis"
                )
                st.caption(f"📈 {len(webcam_detector.compliance_history)} data points")
                st.caption("📊 **Contains:** Timeline data for Excel/Google Sheets")
            except Exception as e:
                st.error(f"CSV export failed: {str(e)}")
                st.caption("⚠️ **Status:** CSV export unavailable")

        with col3:
            st.markdown("##### 📋 **Summary Report**")
            # Create text summary with error handling
            try:
                summary_text = create_text_summary(detailed_stats)
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_text,
                    file_name=f"ppe_detection_summary_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Human-readable executive summary report"
                )
                st.caption("📝 Executive summary")
                st.caption("👔 **Contains:** Business-ready analysis and recommendations")
            except Exception as e:
                st.error(f"Summary export failed: {str(e)}")
                st.caption("⚠️ **Status:** Summary export unavailable")

        # Display live charts
        if webcam_detector.compliance_history:
            st.markdown("#### 📈 **Live Analytics Charts**")

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Compliance timeline chart
                fig_compliance = go.Figure()
                fig_compliance.add_trace(go.Scatter(
                    y=webcam_detector.compliance_history,
                    mode='lines+markers',
                    name='Compliance Rate',
                    line=dict(color='#4CAF50', width=2),
                    marker=dict(size=4)
                ))
                fig_compliance.update_layout(
                    title="Compliance Rate Timeline",
                    yaxis_title="Compliance Rate (%)",
                    xaxis_title="Frame Number",
                    height=300
                )
                st.plotly_chart(fig_compliance, use_container_width=True)

            with col_chart2:
                # People detection chart
                if webcam_detector.people_history:
                    fig_people = go.Figure()
                    fig_people.add_trace(go.Scatter(
                        y=webcam_detector.people_history,
                        mode='lines+markers',
                        name='People Detected',
                        line=dict(color='#2196F3', width=2),
                        marker=dict(size=4)
                    ))
                    fig_people.update_layout(
                        title="People Detection Timeline",
                        yaxis_title="Number of People",
                        xaxis_title="Frame Number",
                        height=300
                    )
                    st.plotly_chart(fig_people, use_container_width=True)

        st.success("✅ Export completed! All files are ready for download.")

    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        st.info("💡 Please try again or contact support if the issue persists.")
        import traceback
        st.code(traceback.format_exc())


# Floating help window functionality removed - help is now in sidebar


# Help window status update function removed - status is now in sidebar


def create_webcam_interface(detection_engine: PPEDetectionEngine, settings: Dict[str, Any]):
    """Create a simplified webcam interface with real-time detection"""

    st.markdown("### 📹 Real-time Detection")

    # Initialize webcam detector with improved error handling
    if 'webcam_detector' not in st.session_state:
        try:
            # Get attendance manager from session state
            attendance_manager = getattr(st.session_state, 'attendance_manager', None)
            st.session_state.webcam_detector = WebcamPPEDetector(detection_engine, attendance_manager)
            st.session_state.analytics_ready = False
            st.session_state.analytics_loading = False
            st.session_state.camera_initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize camera: {str(e)}")
            st.session_state.camera_initialized = False
            return None

    # Check if camera is properly initialized
    if not st.session_state.get('camera_initialized', False):
        st.error("❌ Camera not properly initialized. Please refresh the page.")
        return None

    # Update settings with error handling
    try:
        st.session_state.webcam_detector.update_settings(settings)
    except Exception as e:
        st.warning(f"⚠️ Settings update failed: {str(e)}")

    # Status Dashboard
    status_container = st.container()
    with status_container:
        latest_stats = st.session_state.webcam_detector.get_latest_stats()

        if latest_stats:
            status, status_text, status_desc = get_analytics_status(latest_stats['frame_count'])

            # Create status indicator with time estimates
            if status == "waiting":
                st.info(f"📹 **Camera Status:** Not started")
            elif status == "initializing":
                progress = latest_stats['frame_count'] / 5
                st.warning(f"🔄 **Status:** {status_text}")
                st.progress(progress, text=f"Initializing... {latest_stats['frame_count']}/5 frames")

                # Time estimate for initialization
                if latest_stats['frame_count'] > 1 and latest_stats.get('session_duration', 0) > 0:
                    avg_time_per_frame = latest_stats['session_duration'] / latest_stats['frame_count']
                    remaining_frames = 5 - latest_stats['frame_count']
                    estimated_time = remaining_frames * avg_time_per_frame
                    st.caption(f"⏱️ Estimated time to analytics: {estimated_time:.1f} seconds")

            elif status == "building":
                progress = latest_stats['frame_count'] / 10
                st.info(f"📊 **Status:** {status_text}")
                st.progress(progress, text=f"Building analytics... {latest_stats['frame_count']}/10 frames")

                # Time estimate for full analytics
                if latest_stats['frame_count'] > 5 and latest_stats.get('session_duration', 0) > 0:
                    avg_time_per_frame = latest_stats['session_duration'] / latest_stats['frame_count']
                    remaining_frames = 10 - latest_stats['frame_count']
                    estimated_time = remaining_frames * avg_time_per_frame
                    st.caption(f"⏱️ Full analytics ready in: {estimated_time:.1f} seconds")

            elif status == "ready":
                st.success(f"✅ **Status:** {status_text}")
                st.caption(f"{status_desc} • Session: {format_duration(latest_stats.get('session_duration', 0))}")
            else:  # optimal
                st.success(f"🎯 **Status:** {status_text}")
                st.caption(f"{status_desc} • Session: {format_duration(latest_stats.get('session_duration', 0))}")
        else:
            st.info("📹 **Status:** Waiting for camera to start")

    st.markdown("---")

    # Enhanced CSS for modern UI with glassmorphism and professional styling
    st.markdown("""
    <style>
    /* Modern Theme Variables */
    :root {
        --primary-color: #3B82F6;
        --primary-light: #60A5FA;
        --primary-dark: #1E40AF;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-light: rgba(0, 0, 0, 0.1);
        --shadow-medium: rgba(0, 0, 0, 0.15);
        --shadow-heavy: rgba(0, 0, 0, 0.25);
    }

    /* Modern Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(10px) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px var(--shadow-medium) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Primary Button Glassmorphism */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%) !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        color: white !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-color) 100%) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    }

    /* Secondary Button Glassmorphism */
    .stButton > button[kind="secondary"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(10px) !important;
        color: var(--text-color) !important;
    }

    /* Info/Status Cards */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Checkbox Styling */
    .stCheckbox {
        margin-bottom: 1rem;
    }

    .stCheckbox > label {
        font-weight: 500 !important;
        color: var(--text-color) !important;
    }
    /* Camera Feed Container - Always Full Size */
    .camera-feed-container {
        width: 100%;
        aspect-ratio: 16/9;
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.12),
            0 2px 8px rgba(0,0,0,0.08);
        background: linear-gradient(135deg,
            var(--card-bg, #ffffff) 0%,
            var(--secondary-bg, #f8f9fa) 100%);
        border: 2px solid var(--border-color, #e1e5e9);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .camera-feed-container:hover {
        box-shadow:
            0 12px 40px rgba(0,0,0,0.15),
            0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    /* Camera Placeholder - Full Container Size */
    .camera-placeholder {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg,
            var(--secondary-bg, #f8f9fa) 0%,
            var(--tertiary-bg, #e9ecef) 100%);
        color: var(--text-color, #495057);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        z-index: 1;
    }

    .camera-placeholder-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.6;
        animation: pulse 2s infinite;
    }

    .camera-placeholder-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .camera-placeholder-subtitle {
        font-size: 1rem;
        opacity: 0.7;
        text-align: center;
        max-width: 300px;
        line-height: 1.4;
    }

    /* State Badge */
    .camera-state-badge {
        position: absolute;
        top: 16px;
        left: 16px;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        z-index: 10;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }

    .state-inactive {
        background: rgba(244, 67, 54, 0.9);
        color: white;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
    }

    .state-active {
        background: rgba(76, 175, 80, 0.9);
        color: white;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        animation: pulse-green 2s infinite;
    }

    .state-paused {
        background: rgba(255, 152, 0, 0.9);
        color: white;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.3);
    }

    /* WebRTC Video Styling */
    .camera-feed-container video {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover;
        border-radius: 14px;
        position: relative;
        z-index: 2;
    }

    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 0.3; }
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3); }
        50% { box-shadow: 0 4px 20px rgba(76, 175, 80, 0.5); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .camera-feed-container {
            border-radius: 12px;
        }

        .camera-placeholder-icon {
            font-size: 3rem;
        }

        .camera-placeholder-text {
            font-size: 1.2rem;
        }

        .camera-placeholder-subtitle {
            font-size: 0.9rem;
        }

        .camera-state-badge {
            top: 12px;
            left: 12px;
            padding: 6px 12px;
            font-size: 0.8rem;
        }

        /* Mobile Layout Adjustments */
        .stColumns {
            flex-direction: column !important;
        }

        .stButton > button {
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
        }
    }

    /* Tablet Responsive Design */
    @media (max-width: 1024px) {
        .camera-feed-container {
            border-radius: 14px;
        }

        .camera-placeholder-icon {
            font-size: 3.5rem;
        }

        .camera-placeholder-text {
            font-size: 1.3rem;
        }
    }

    /* Camera Quality Controls Styling */
    .stExpander > div > div > div > div {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }

    .stExpander > div > div > div > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Checkbox enhancements */
    .stCheckbox > label {
        font-weight: 500;
        color: #2c3e50;
        transition: color 0.3s ease;
    }

    .stCheckbox > label:hover {
        color: #667eea;
    }

    /* Large Screen Optimizations */
    @media (min-width: 1200px) {
        .camera-feed-container {
            border-radius: 20px;
        }

        .camera-placeholder-icon {
            font-size: 5rem;
        }

        .camera-placeholder-text {
            font-size: 1.8rem;
        }

        .camera-state-badge {
            top: 20px;
            left: 20px;
            padding: 10px 20px;
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Responsive layout: Desktop 65/35 split, Mobile stacked
    # Check if mobile view (simplified detection)
    is_mobile = st.session_state.get('mobile_view', False)

    # Add mobile toggle for testing
    if st.sidebar.button("📱 Toggle Mobile View"):
        st.session_state.mobile_view = not st.session_state.get('mobile_view', False)
        st.rerun()

    # Responsive column layout
    if is_mobile:
        # Mobile: Single column layout
        col_camera = st.container()
        col_stats = st.container()
    else:
        # Desktop: Two-column layout (65% camera, 35% stats)
        col_camera, col_stats = st.columns([0.65, 0.35], gap="large")

    with col_camera:


        # Enhanced Camera quality indicator with dynamic settings
        camera_quality = settings.get('camera_quality', 'HD')
        camera_icon = settings.get('camera_icon', '🟢')
        camera_width = settings.get('camera_width', 1280)
        camera_height = settings.get('camera_height', 720)

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            <div style="font-size: 1.1rem; margin-bottom: 5px;">
                {camera_icon} Camera Quality: {camera_quality}
            </div>
            <div style="font-size: 0.85rem; opacity: 0.9;">
                Resolution: {camera_width}×{camera_height} • Enhanced Clarity
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Camera Quality Controls in Main Area (Collapsible)
        with st.expander("📹 Camera Quality Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                # Quick quality presets - sync with session state
                quality_options = ["Standard (480p)", "HD (720p)", "Full HD (1080p)", "Ultra HD (1440p)"]
                current_selection = st.session_state.get('camera_quality_selection', 'HD (720p)')
                current_index = quality_options.index(current_selection) if current_selection in quality_options else 1

                quality_preset = st.selectbox(
                    "Quality Preset:",
                    quality_options,
                    index=current_index,
                    key="main_camera_quality"
                )

                # Update settings based on selection
                quality_map = {
                    "Standard (480p)": {"width": 640, "height": 480, "label": "Standard", "icon": "🟡"},
                    "HD (720p)": {"width": 1280, "height": 720, "label": "HD", "icon": "🟢"},
                    "Full HD (1080p)": {"width": 1920, "height": 1080, "label": "Full HD", "icon": "🔵"},
                    "Ultra HD (1440p)": {"width": 2560, "height": 1440, "label": "Ultra HD", "icon": "🟣"}
                }

                if quality_preset in quality_map:
                    quality_config = quality_map[quality_preset]
                    # Update the settings with new quality
                    settings['camera_quality'] = quality_config['label']
                    settings['camera_width'] = quality_config['width']
                    settings['camera_height'] = quality_config['height']
                    settings['camera_icon'] = quality_config['icon']

                    # Update session state to sync with other components
                    st.session_state.camera_quality_selection = quality_preset

                    # Update the variables for WebRTC
                    camera_quality = quality_config['label']
                    camera_icon = quality_config['icon']
                    camera_width = quality_config['width']
                    camera_height = quality_config['height']

                    # Show success message for quality change
                    st.success(f"📹 Camera quality updated to {quality_config['label']} ({quality_config['width']}×{quality_config['height']})")
                    time.sleep(0.5)  # Brief pause to show the message

            with col2:
                # Camera enhancement options
                st.markdown("**Enhancement Options:**")
                auto_focus = st.checkbox("🎯 Auto Focus", value=True, help="Enable automatic focus adjustment")
                noise_reduction = st.checkbox("🔇 Noise Reduction", value=True, help="Reduce camera noise for clearer image")

                # Add these to settings
                settings['auto_focus'] = auto_focus
                settings['noise_reduction'] = noise_reduction

                # Quality info
                st.info(f"📊 Current: {camera_width}×{camera_height}")

        # Speed Settings Controls in Main Area (Collapsible)
        with st.expander("⚡ Speed & Detection Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                # Speed presets
                speed_preset = st.selectbox(
                    "Speed Mode:",
                    ["🚀 Ultra Fast", "⚡ Fast", "🎯 Balanced", "🔍 Accurate"],
                    index=0,
                    key="main_speed_preset",
                    help="Choose processing speed vs accuracy balance"
                )

                # Map presets to settings
                preset_map = {
                    "🚀 Ultra Fast": {"conf": 0.6, "iou": 0.5, "skip": 4, "res": 640},
                    "⚡ Fast": {"conf": 0.5, "iou": 0.45, "skip": 3, "res": 640},
                    "🎯 Balanced": {"conf": 0.5, "iou": 0.45, "skip": 2, "res": 1280},
                    "🔍 Accurate": {"conf": 0.4, "iou": 0.4, "skip": 1, "res": 1280}
                }

                # Update settings with speed preset
                speed_settings = preset_map[speed_preset]
                settings.update(speed_settings)

                # Speed info display
                speed_multiplier = speed_settings['skip']
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 10px 0;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 0.9rem;
                ">
                    🚀 {speed_multiplier}x Speed Boost<br>
                    <small style="opacity: 0.9;">Resolution: {speed_settings['res']}p</small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Advanced detection settings
                st.markdown("**Detection Settings:**")
                conf_threshold = st.slider("Confidence", 0.1, 1.0, speed_settings['conf'], 0.05, key="main_conf")
                iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, speed_settings['iou'], 0.05, key="main_iou")

                # Update settings with manual adjustments
                settings['conf'] = conf_threshold
                settings['iou'] = iou_threshold

                # Detection info
                st.info(f"🎯 Confidence: {conf_threshold:.2f} | IoU: {iou_threshold:.2f}")

        # Detection toggle with better styling
        detection_enabled = st.checkbox(
            "🔍 Enable PPE Detection",
            value=True,
            help="Toggle real-time PPE detection analysis"
        )

        # Full-size camera container with state management
        camera_active = False
        if 'webrtc_ctx' in locals():
            camera_active = webrtc_ctx.state.playing if webrtc_ctx else False

        # Determine camera state for badge
        if camera_active:
            state_class = "state-active"
            state_text = "🟢 Live"
            state_subtitle = "Camera is active and streaming"
        else:
            state_class = "state-inactive"
            state_text = "🔴 Inactive"
            state_subtitle = "Click START to begin detection"

        # Camera feed container with placeholder
        st.markdown(f"""
        <div class="camera-feed-container">
            <div class="camera-state-badge {state_class}">
                {state_text}
            </div>
            <div class="camera-placeholder" id="camera-placeholder">
                <div class="camera-placeholder-icon">📹</div>
                <div class="camera-placeholder-text">Camera Feed</div>
                <div class="camera-placeholder-subtitle">{state_subtitle}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # WebRTC configuration with optimized settings
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        # Enhanced WebRTC configuration with dynamic quality and camera enhancements
        video_constraints = {
            "width": {"ideal": camera_width, "max": max(camera_width, 2560)},
            "height": {"ideal": camera_height, "max": max(camera_height, 1440)},
            "frameRate": {"ideal": 25, "max": 30},
            "aspectRatio": camera_width / camera_height
        }

        # Add enhancement settings if available
        if settings.get('auto_focus', True):
            video_constraints["focusMode"] = "continuous"

        if settings.get('noise_reduction', True):
            video_constraints["noiseSuppression"] = True
            video_constraints["echoCancellation"] = True

        # Create webrtc streamer with enhanced settings
        webrtc_ctx = webrtc_streamer(
            key="ppe-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=st.session_state.webcam_detector.video_frame_callback,
            media_stream_constraints={
                "video": video_constraints,
                "audio": False
            },
            async_processing=True,
        )

        # Update detection settings
        st.session_state.webcam_detector.update_settings({
            **settings,
            'detection_enabled': detection_enabled
        })





    # Stats section - responsive handling
    if is_mobile:
        st.markdown("---")

    with col_stats:
        # Live Stats Section with Enhanced Header
        if is_mobile:
            # Mobile: Collapsible stats section
            with st.expander("📊 Live Analytics", expanded=True):
                stats_content = st.container()
        else:
            # Desktop: Always visible stats
            st.markdown("""
            <div style="margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: var(--text-color); font-weight: 600;">
                    📊 Live Analytics
                </h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.9rem;">
                    Real-time detection statistics
                </p>
            </div>
            """, unsafe_allow_html=True)
            stats_content = st.container()

        with stats_content:
            # Enhanced state indicator for analytics
            camera_active = webrtc_ctx and webrtc_ctx.state.playing
            analytics_paused = st.session_state.get('analytics_paused', False)

            # Track camera state changes and manage session accordingly
            if 'previous_camera_state' not in st.session_state:
                st.session_state.previous_camera_state = False

            # Detect camera state changes
            if camera_active != st.session_state.previous_camera_state:
                if camera_active:
                    # Camera just started
                    st.session_state.webcam_detector.start_session()
                else:
                    # Camera just stopped
                    st.session_state.webcam_detector.stop_session()
                st.session_state.previous_camera_state = camera_active

            # Determine analytics state
            if camera_active and not analytics_paused:
                analytics_state = "active"
                analytics_badge = "🟢 Active"
                analytics_color = "#4CAF50"
            elif camera_active and analytics_paused:
                analytics_state = "paused"
                analytics_badge = "🟡 Paused"
                analytics_color = "#FF9800"
            else:
                analytics_state = "inactive"
                analytics_badge = "🔴 Inactive"
                analytics_color = "#F44336"

            # Analytics status card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {analytics_color}15 0%, {analytics_color}05 100%);
                border: 2px solid {analytics_color}30;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1.5rem;
                text-align: center;
            ">
                <div style="
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: {analytics_color};
                    margin-bottom: 0.5rem;
                ">
                    {analytics_badge}
                </div>
                <div style="
                    font-size: 0.9rem;
                    opacity: 0.8;
                    color: var(--text-color);
                ">
                    {'Real-time analytics running' if analytics_state == 'active'
                     else 'Analytics temporarily paused' if analytics_state == 'paused'
                     else 'Start camera to begin analytics'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Add Control Buttons Section
            st.markdown("---")

            # Add CSS styling for control buttons
            st.markdown("""
            <style>
            /* Control Panel Styling */
            .control-buttons-section {
                background: linear-gradient(135deg, rgba(77, 171, 247, 0.08) 0%, rgba(51, 154, 240, 0.08) 100%);
                padding: 1.5rem;
                border-radius: 16px;
                border: 2px solid rgba(77, 171, 247, 0.2);
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(77, 171, 247, 0.1);
                position: relative;
                overflow: hidden;
            }

            .control-buttons-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #4dabf7, #339af0, #4dabf7);
                border-radius: 16px 16px 0 0;
            }

            /* Enhanced button styling for control panel */
            .control-buttons-section .stButton > button {
                font-size: 0.9rem !important;
                padding: 0.7rem 1rem !important;
                height: auto !important;
                min-height: 2.8rem !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            }

            .control-buttons-section .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
            }

            .control-buttons-section .stButton > button:active {
                transform: translateY(0px) !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            }

            /* Responsive design for control buttons */
            @media (max-width: 768px) {
                .control-buttons-section {
                    padding: 1rem;
                    margin: 0.5rem 0;
                }

                .control-buttons-section .stButton > button {
                    font-size: 0.8rem !important;
                    padding: 0.6rem 0.8rem !important;
                    min-height: 2.5rem !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="control-buttons-section">
                <div style="margin-bottom: 1rem; text-align: center;">
                    <h4 style="margin: 0; color: var(--text-color); font-weight: 600;">
                        🎛️ Control Panel
                    </h4>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.9rem;">
                        Camera and analytics controls
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Get webcam status for button states
            webcam_active = webrtc_ctx and webrtc_ctx.state.playing

            # Get fresh stats for export check
            current_stats = st.session_state.webcam_detector.get_latest_stats()
            has_stats = current_stats and current_stats.get('frame_count', 0) > 0
            has_frames = st.session_state.webcam_detector.frame_count > 0

            # Primary Controls Row
            col_ctrl1, col_ctrl2 = st.columns(2)

            with col_ctrl1:
                # Enhanced Reset All button - works regardless of camera state
                if st.button("🔄 Reset",
                            use_container_width=True,
                            help="Reset all statistics, clear session data, and restart the entire detection process",
                            type="primary"):
                    try:
                        # Complete reset of all detection processes
                        if hasattr(st.session_state, 'webcam_detector'):
                            st.session_state.webcam_detector.reset_stats()

                        # Clear all session states related to webcam detection
                        keys_to_clear = [
                            'webcam_detector', 'analytics_ready', 'analytics_loading',
                            'last_stats_update', 'analytics_paused', 'manual_refresh',
                            'webcam_export_data', 'reset_success_time', 'stats_loading'
                        ]

                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]

                        st.success("✅ All processes reset successfully! Please restart the camera.")
                        time.sleep(1)  # Give user time to see the message
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Reset failed: {str(e)}")

            with col_ctrl2:
                # Manual refresh button with enhanced functionality
                refresh_disabled = not webcam_active
                refresh_help = "Force immediate refresh of live statistics display" if not refresh_disabled else "⚠️ Start camera first to refresh"

                if st.button("📊 Refresh",
                            use_container_width=True,
                            help=refresh_help,
                            disabled=refresh_disabled,
                            type="secondary"):
                    try:
                        st.session_state.last_stats_update = time.time() - 2  # Force immediate update
                        st.session_state.manual_refresh = True
                        st.info("🔄 Live statistics refreshed!")
                        time.sleep(0.3)  # Brief pause to show message
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Refresh failed: {str(e)}")

            # Secondary Controls Row
            col_ctrl3, col_ctrl4 = st.columns(2)

            with col_ctrl3:
                # Pause/Resume analytics with enhanced functionality
                if 'analytics_paused' not in st.session_state:
                    st.session_state.analytics_paused = False

                pause_disabled = not webcam_active

                if st.session_state.analytics_paused:
                    resume_help = "Resume analytics updates" if not pause_disabled else "⚠️ Start camera first"
                    if st.button("▶️ Resume",
                               use_container_width=True,
                               help=resume_help,
                               disabled=pause_disabled,
                               type="primary"):
                        st.session_state.analytics_paused = False
                        st.session_state.last_stats_update = time.time() - 1  # Force immediate update
                        st.success("▶️ Analytics resumed!")
                        time.sleep(0.3)
                        st.rerun()
                else:
                    pause_help = "Pause analytics updates" if not pause_disabled else "⚠️ Start camera first"
                    if st.button("⏸️ Pause",
                               use_container_width=True,
                               help=pause_help,
                               disabled=pause_disabled,
                               type="secondary"):
                        st.session_state.analytics_paused = True
                        st.warning("⏸️ Analytics paused!")
                        time.sleep(0.3)
                        st.rerun()

            with col_ctrl4:
                # Enhanced Export functionality with better data validation
                export_disabled = not (has_stats or has_frames)

                # Better export help text
                if export_disabled:
                    if not webcam_active:
                        export_help = "⚠️ Start camera first to enable export"
                    elif st.session_state.webcam_detector.frame_count == 0:
                        export_help = "⚠️ No frames processed yet - wait for detection to start"
                    else:
                        export_help = "⚠️ Analytics not ready - wait a few more seconds"
                else:
                    frame_count = st.session_state.webcam_detector.frame_count
                    export_help = f"Export session data ({frame_count:,} frames processed)"

                if st.button("📊 Go to Export Tab",
                            use_container_width=True,
                            help=export_help,
                            disabled=export_disabled,
                            type="primary"):

                    # Show floating animated popup notification
                    show_export_popup_notification()

            # Show confirmation message if user confirmed the popup
            if st.session_state.get('export_popup_confirmed', False):
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                    color: white;
                    padding: 1rem 1.5rem;
                    border-radius: 12px;
                    margin: 1rem 0;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
                    animation: slideInUp 0.5s ease-out;
                    border-left: 4px solid #2E7D32;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                        ✅ Export Ready!
                    </div>
                    <div style="font-size: 1rem; opacity: 0.9;">
                        Please click on the <strong>📊 Export Session Data</strong> tab above to access your export interface.
                    </div>
                    <div style="margin-top: 0.8rem; font-size: 0.9rem; opacity: 0.8;">
                        💡 Enhanced download options, analytics charts, and detailed reports await!
                    </div>
                </div>

                <style>
                @keyframes slideInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                </style>
                """, unsafe_allow_html=True)

                # Add a dismiss button for the confirmation message
                if st.button("✕ Dismiss", key="dismiss_export_confirmation", help="Dismiss this message"):
                    st.session_state.export_popup_confirmed = False
                    st.rerun()

            # Close control buttons section
            st.markdown("</div>", unsafe_allow_html=True)

            # Create stats container
            stats_container = st.container()

        # Real-time statistics with enhanced state management
        if camera_active:
            # Auto-refresh every 1 second for better performance
            if 'last_stats_update' not in st.session_state:
                st.session_state.last_stats_update = time.time()
                st.session_state.stats_loading = True

            current_time = time.time()

            # Get latest stats with better error handling
            try:
                latest_stats = st.session_state.webcam_detector.get_latest_stats()
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
                latest_stats = None

            # Check if analytics are ready (need at least 3 frames for meaningful stats)
            if latest_stats and latest_stats['frame_count'] >= 3:
                st.session_state.analytics_ready = True
                st.session_state.analytics_loading = False
            elif latest_stats and latest_stats['frame_count'] > 0:
                st.session_state.analytics_loading = True
                st.session_state.analytics_ready = False

            # Auto-refresh logic - improved timing
            should_auto_refresh = (
                current_time - st.session_state.last_stats_update > 1.0 and
                not st.session_state.get('analytics_paused', False) and
                not st.session_state.get('manual_refresh', False)
            )

            if should_auto_refresh:
                st.session_state.last_stats_update = current_time
                st.rerun()

            # Clear manual refresh flag after one cycle
            if st.session_state.get('manual_refresh', False):
                st.session_state.manual_refresh = False

            with stats_container:
                if latest_stats and latest_stats['frame_count'] > 0:
                    # Show loading progress for analytics
                    if st.session_state.analytics_loading and not st.session_state.analytics_ready:
                        frames_needed = 3
                        current_frames = latest_stats['frame_count']
                        progress = min(current_frames / frames_needed, 1.0)

                        st.info("🔄 **Initializing Analytics...**")
                        st.progress(progress, text=f"Processing frames: {current_frames}/{frames_needed}")
                        st.caption(f"⏱️ Analytics will be ready in {max(0, frames_needed - current_frames)} more frames")

                        # Show basic stats while loading
                        st.markdown("**Current Frame:**")
                        st.write(f"👥 People: **{latest_stats.get('total_people', 0)}**")
                        st.write(f"⚠️ Violations: **{latest_stats.get('violations', 0)}**")
                        st.write(f"📊 Compliance: **{latest_stats.get('compliance_rate', 0):.0f}%**")

                    else:
                        # Full analytics ready - show complete metrics
                        if st.session_state.analytics_ready:
                            st.success("✅ **Live Analytics Active!**")

                        # Enhanced metrics with real-time accuracy and better formatting
                        col_a, col_b = st.columns(2)

                        with col_a:
                            # Current frame people count with session context
                            people_count = latest_stats.get('total_people', 0)
                            avg_people = latest_stats.get('avg_people', 0)
                            people_delta = f"Avg: {avg_people:.1f}" if avg_people > 0 else None

                            st.metric(
                                label="👥 People (Current)",
                                value=people_count,
                                delta=people_delta,
                                help="Number of people detected in current frame vs session average"
                            )

                            # Current compliance with trend indicator
                            compliance_rate = latest_stats.get('compliance_rate', 0)
                            avg_compliance = latest_stats.get('avg_compliance_rate', 0)

                            if avg_compliance > 0:
                                compliance_trend = compliance_rate - avg_compliance
                                if abs(compliance_trend) > 5:  # Only show significant changes
                                    compliance_delta = f"{compliance_trend:+.0f}% vs avg"
                                else:
                                    compliance_delta = "Stable"
                            else:
                                compliance_delta = None

                            st.metric(
                                label="📊 Compliance (Current)",
                                value=f"{compliance_rate:.0f}%",
                                delta=compliance_delta,
                                help="Current frame PPE compliance vs session average"
                            )

                        with col_b:
                            # Current violations with session context
                            violations = latest_stats.get('violations', 0)
                            total_violations = latest_stats.get('total_violations_session', 0)
                            violation_delta = f"Total: {total_violations}" if total_violations > 0 else "None detected"

                            st.metric(
                                label="⚠️ Violations (Current)",
                                value=violations,
                                delta=violation_delta,
                                delta_color="inverse" if violations > 0 else "normal",
                                help="Current frame violations vs total session violations"
                            )

                            # Accurate FPS calculation with performance indicator
                            try:
                                session_duration = latest_stats.get('session_duration', 0)
                                if session_duration > 0:
                                    fps = latest_stats['frame_count'] / session_duration

                                    # Performance indicator
                                    if fps >= 15:
                                        fps_delta = "Excellent"
                                    elif fps >= 10:
                                        fps_delta = "Good"
                                    elif fps >= 5:
                                        fps_delta = "Fair"
                                    else:
                                        fps_delta = "Slow"

                                    st.metric(
                                        label="🎥 Processing FPS",
                                        value=f"{fps:.1f}",
                                        delta=fps_delta,
                                        help="Average frames processed per second"
                                    )
                                else:
                                    st.metric(label="🎥 Processing FPS", value="0.0", help="Calculating...")
                            except Exception:
                                st.metric(label="🎥 Processing FPS", value="N/A", help="Error calculating FPS")

                        # Enhanced status indicator with better styling
                        st.markdown("---")
                        if compliance_rate >= 90:
                            st.success("🟢 **EXCELLENT** - All PPE requirements met!")
                        elif compliance_rate >= 75:
                            st.success("✅ **GOOD** - Most PPE requirements met")
                        elif compliance_rate >= 50:
                            st.warning("⚠️ **FAIR** - Some PPE violations detected")
                        else:
                            st.error("❌ **POOR** - Multiple PPE violations!")

                        # Session information
                        st.markdown("---")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.caption(f"🎬 Frame: {latest_stats['frame_count']:,}")
                        with col_info2:
                            session_duration = latest_stats.get('session_duration', 0)
                            st.caption(f"⏱️ Session: {format_duration(session_duration)}")

                        # Enhanced session summary with accuracy metrics
                        if latest_stats['frame_count'] >= 5:
                            st.markdown("---")
                            st.markdown("**📊 Real-Time Session Summary:**")

                            # Main session metrics
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                avg_compliance = latest_stats.get('avg_compliance_rate', 0)
                                st.write(f"📈 **Session Avg Compliance:** {avg_compliance:.1f}%")
                                total_people = latest_stats.get('total_people_session', 0)
                                st.write(f"👥 **Total People Detected:** {total_people:,}")

                            with col_s2:
                                total_violations = latest_stats.get('total_violations_session', 0)
                                st.write(f"⚠️ **Total Violations:** {total_violations:,}")
                                session_duration = latest_stats.get('session_duration', 0)
                                if session_duration > 0:
                                    avg_fps = latest_stats['frame_count'] / session_duration
                                    st.write(f"🎥 **Avg Processing FPS:** {avg_fps:.1f}")

                            with col_s3:
                                # Detection accuracy metrics
                                detection_rate = latest_stats.get('detection_rate', 0)
                                st.write(f"🎯 **Detection Rate:** {detection_rate:.1f}%")
                                frames_with_people = latest_stats.get('frames_with_people', 0)
                                frame_count = latest_stats['frame_count']
                                people_frame_ratio = (frames_with_people / frame_count * 100) if frame_count > 0 else 0
                                st.write(f"👥 **Frames with People:** {people_frame_ratio:.1f}%")

                            # Session quality indicator
                            st.markdown("---")
                            if avg_compliance >= 90:
                                st.success(f"🟢 **Excellent Session Quality** - {avg_compliance:.1f}% compliance maintained")
                            elif avg_compliance >= 75:
                                st.info(f"✅ **Good Session Quality** - {avg_compliance:.1f}% compliance maintained")
                            elif avg_compliance >= 50:
                                st.warning(f"⚠️ **Fair Session Quality** - {avg_compliance:.1f}% compliance - room for improvement")
                            else:
                                st.error(f"❌ **Poor Session Quality** - {avg_compliance:.1f}% compliance - immediate attention needed")

                else:
                    # Better debugging and status information
                    detector = st.session_state.webcam_detector

                    # Check if detection is actually enabled
                    if not detection_enabled:
                        st.warning("⚠️ **Detection Disabled**")
                        st.caption("Enable PPE Detection checkbox to start analysis")
                    else:
                        # Check frame processing status
                        if detector.frame_count == 0:
                            st.info("📊 **Initializing Detection...**")
                            st.caption("📹 Waiting for camera frames...")

                            # Show helpful tips
                            with st.expander("💡 Troubleshooting Tips", expanded=False):
                                st.markdown("""
                                **If detection doesn't start:**
                                1. ✅ Ensure camera permissions are granted
                                2. 🔄 Try refreshing the page
                                3. 📹 Check if camera is working in other apps
                                4. 🔍 Make sure 'Enable PPE Detection' is checked
                                5. ⏱️ Wait a few seconds for initialization
                                """)
                        else:
                            st.info("🔄 **Processing Frames...**")
                            st.caption(f"📊 Processed {detector.frame_count} frames, building analytics...")

                            # Show basic frame info
                            st.progress(min(detector.frame_count / 3, 1.0),
                                      text=f"Analytics ready in {max(0, 3 - detector.frame_count)} frames")
        else:
            with stats_container:
                # Show last known stats if available
                if hasattr(st.session_state, 'webcam_detector'):
                    latest_stats = st.session_state.webcam_detector.get_latest_stats()
                    if latest_stats and latest_stats['frame_count'] > 0:
                        st.markdown("---")
                        st.markdown("**Last Session:**")
                        st.caption(f"Processed {latest_stats['frame_count']} frames")
                        st.caption(f"Final compliance: {latest_stats['compliance_rate']:.0f}%")





    # Help information is now available in the sidebar

    return webrtc_ctx


def create_webcam_analytics(webcam_detector: WebcamPPEDetector):
    """Create comprehensive analytics for webcam detection session with improved data handling"""

    st.markdown("#### 📈 **Session Analytics Dashboard**")

    # Get session statistics with error handling
    try:
        latest_stats = webcam_detector.get_latest_stats()
    except Exception as e:
        st.error(f"Error retrieving analytics: {str(e)}")
        return

    # Check analytics readiness and show loading if needed
    if not latest_stats or latest_stats['frame_count'] == 0:
        # No data yet - show waiting state
        st.info("📊 **Waiting for camera to start...**")

        # Show preview of what will be available
        with st.expander("📋 **Preview: Available Analytics**", expanded=True):
            st.markdown("""
            **📊 Real-time Monitoring:**
            - 👥 People detection count and trends
            - ⚠️ PPE violation tracking and alerts
            - 📈 Live compliance rate calculation
            - 🎥 Processing performance metrics

            **📈 Session Statistics:**
            - ⏱️ Session duration and frame count
            - 📊 Average compliance over time
            - 🎯 Detection consistency analysis
            - ⚡ Real-time processing performance

            **📋 Performance Indicators:**
            - ✅ Detection accuracy rates
            - 🔄 Frame processing speed (FPS)
            - 📊 Compliance trend analysis
            - 🎯 System performance metrics
            """)
        return

    elif latest_stats['frame_count'] < 5:
        # Loading phase - show progress with better UX
        frames_for_analytics = 5
        current_frames = latest_stats['frame_count']
        progress = current_frames / frames_for_analytics

        st.warning("🔄 **Initializing Analytics...**")

        # Enhanced progress display
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.progress(progress, text=f"Processing frame {current_frames}/{frames_for_analytics}")
        with progress_col2:
            st.metric("Progress", f"{progress*100:.0f}%")

        # Time estimation with better accuracy
        if current_frames > 1:
            session_duration = latest_stats.get('session_duration', 0)
            if session_duration > 0:
                avg_time_per_frame = session_duration / current_frames
                remaining_frames = frames_for_analytics - current_frames
                estimated_time = remaining_frames * avg_time_per_frame
                st.caption(f"⏱️ Estimated time remaining: {estimated_time:.1f} seconds")
            else:
                st.caption("⏱️ Calculating time estimate...")
        else:
            st.caption("⏱️ Initializing timing calculations...")

        # Show current frame data while loading
        st.markdown("---")
        st.markdown("**📊 Current Frame Data:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"👥 People: **{latest_stats.get('total_people', 0)}**")
            st.write(f"📊 Compliance: **{latest_stats.get('compliance_rate', 0):.0f}%**")
        with col_b:
            st.write(f"⚠️ Violations: **{latest_stats.get('violations', 0)}**")
            if latest_stats.get('session_duration', 0) > 0:
                fps = current_frames / latest_stats['session_duration']
                st.write(f"🎥 FPS: **{fps:.1f}**")
            else:
                st.write(f"🎥 FPS: **Calculating...**")

        return

    elif latest_stats['frame_count'] >= 5:
        # Analytics ready - show full comprehensive report
        st.success("✅ **Analytics Dashboard Active!**")

        # Enhanced data quality indicator
        data_quality = min(100, (latest_stats['frame_count'] / 30) * 100)
        quality_col1, quality_col2 = st.columns([3, 1])

        with quality_col1:
            if data_quality >= 100:
                st.info("🎯 **High Quality Data** - Comprehensive analytics with high confidence")
            elif data_quality >= 50:
                st.info("📊 **Good Quality Data** - Reliable analytics with good confidence")
            else:
                st.warning("⚠️ **Basic Quality Data** - Limited analytics, continue for better quality")

        with quality_col2:
            st.metric("Data Quality", f"{data_quality:.0f}%")

    if latest_stats and latest_stats['frame_count'] >= 5:
        # Enhanced main metrics row with better formatting
        st.markdown("---")
        st.markdown("##### 📊 **Key Session Metrics**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            frame_count = latest_stats['frame_count']
            st.metric(
                "📊 Total Frames",
                f"{frame_count:,}",
                delta=f"+{frame_count}" if frame_count > 0 else None,
                help="Total number of frames processed in this session"
            )

        with col2:
            duration = latest_stats.get('session_duration', 0)
            if duration > 0:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes:02d}:{seconds:02d}"
            else:
                duration_str = "00:00"
            st.metric(
                "⏱️ Duration",
                duration_str,
                help="Duration of current detection session"
            )

        with col3:
            if duration > 0:
                fps = latest_stats['frame_count'] / duration
                st.metric(
                    "🎥 Processing FPS",
                    f"{fps:.1f}",
                    delta=f"{fps - 15:.1f}" if fps > 0 else None,
                    help="Average frames processed per second"
                )
            else:
                st.metric("🎥 Processing FPS", "0.0", help="Average frames processed per second")

        with col4:
            avg_compliance = latest_stats.get('avg_compliance_rate', 0)
            compliance_delta = None
            if avg_compliance > 0:
                if avg_compliance >= 90:
                    compliance_delta = "Excellent!"
                elif avg_compliance >= 75:
                    compliance_delta = "Good"
                elif avg_compliance >= 50:
                    compliance_delta = "Fair"
                else:
                    compliance_delta = "Needs Improvement"

            st.metric(
                "📈 Avg Compliance",
                f"{avg_compliance:.1f}%",
                delta=compliance_delta,
                help="Average compliance rate over the session"
            )

        # Detailed analytics
        st.markdown("---")
        st.markdown("##### 🔍 Detailed Statistics")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**👥 People Detection:**")
            total_people = latest_stats.get('total_people_session', 0)
            avg_people = latest_stats.get('avg_people', 0)

            st.write(f"• Total people detected: **{total_people}**")
            st.write(f"• Average per frame: **{avg_people:.1f}**")
            st.write(f"• Current frame: **{latest_stats['total_people']}**")

            # People detection rate
            if latest_stats['frame_count'] > 0:
                people_rate = (total_people / latest_stats['frame_count']) * 100
                st.progress(min(people_rate / 100, 1.0))
                st.caption(f"People detection rate: {people_rate:.1f}%")

        with col_b:
            st.markdown("**⚠️ Violation Analysis:**")
            total_violations = latest_stats.get('total_violations_session', 0)
            avg_violations = latest_stats.get('avg_violations', 0)

            st.write(f"• Total violations: **{total_violations}**")
            st.write(f"• Average per frame: **{avg_violations:.1f}**")
            st.write(f"• Current frame: **{latest_stats['violations']}**")

            # Violation rate
            if total_people > 0:
                violation_rate = (total_violations / total_people) * 100
                st.progress(min(violation_rate / 100, 1.0))
                st.caption(f"Violation rate: {violation_rate:.1f}%")
            else:
                st.caption("No people detected yet")

        # Performance indicators
        st.markdown("---")
        st.markdown("##### 🎯 Performance Indicators")

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            # Detection consistency
            if len(webcam_detector.compliance_history) > 5:
                consistency = 100 - (np.std(webcam_detector.compliance_history) * 2)
                consistency = max(0, min(100, consistency))
                st.metric("🎯 Consistency", f"{consistency:.0f}%", help="How consistent the compliance rate is")
            else:
                st.metric("🎯 Consistency", "N/A", help="Need more frames for calculation")

        with perf_col2:
            # Real-time performance
            if duration > 0:
                real_time_ratio = fps / 30  # Assuming 30 FPS target
                performance = min(100, real_time_ratio * 100)
                st.metric("⚡ Real-time", f"{performance:.0f}%", help="Processing speed vs real-time")
            else:
                st.metric("⚡ Real-time", "N/A")

        with perf_col3:
            # Detection accuracy (simplified metric)
            if latest_stats['frame_count'] > 0:
                accuracy = min(100, (latest_stats['frame_count'] / max(1, latest_stats['frame_count'])) * 100)
                st.metric("✅ Detection Rate", f"{accuracy:.0f}%", help="Successful detection rate")
            else:
                st.metric("✅ Detection Rate", "N/A")

    else:
        st.info("📊 Start webcam detection to see detailed analytics")
        st.markdown("""
        **Available Analytics:**
        - Real-time compliance monitoring
        - People detection statistics
        - Violation tracking and trends
        - Performance metrics
        - Session duration and FPS
        """)




# Demo function removed - use main app instead
