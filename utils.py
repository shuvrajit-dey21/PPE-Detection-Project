"""
Utility functions for PPE Compliance Monitoring System
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json


def load_css(file_path: str = None, css_string: str = None):
    """Load custom CSS for styling the Streamlit app
    
    Args:
        file_path: Path to CSS file
        css_string: CSS string to apply directly
    """
    if file_path and os.path.exists(file_path):
        with open(file_path) as f:
            css_string = f.read()
    
    if css_string:
        st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


def get_base64_image(image_path: str) -> str:
    """Convert image to base64 string for embedding in HTML
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""


def create_download_link(data: bytes, filename: str, text: str = "Download") -> str:
    """Create a download link for binary data
    
    Args:
        data: Binary data to download
        filename: Name of the file
        text: Link text
        
    Returns:
        HTML download link
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href


def format_detection_results(detections: List[Dict]) -> pd.DataFrame:
    """Format detection results into a pandas DataFrame
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Formatted DataFrame
    """
    if not detections:
        return pd.DataFrame()
    
    formatted_data = []
    for i, det in enumerate(detections):
        formatted_data.append({
            'ID': i + 1,
            'Class': det['class_name'],
            'Confidence': f"{det['confidence']:.3f}",
            'Bbox (x1,y1,x2,y2)': f"({det['bbox'][0]:.0f},{det['bbox'][1]:.0f},{det['bbox'][2]:.0f},{det['bbox'][3]:.0f})",
            'Compliant': '✅' if det.get('is_compliant') else '❌' if det.get('is_compliant') is False else '➖'
        })
    
    return pd.DataFrame(formatted_data)


def create_compliance_chart(compliance_data: List[float], title: str = "Compliance Over Time") -> go.Figure:
    """Create a compliance rate chart
    
    Args:
        compliance_data: List of compliance rates
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add compliance line
    fig.add_trace(go.Scatter(
        x=list(range(len(compliance_data))),
        y=compliance_data,
        mode='lines+markers',
        name='Compliance Rate',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Add threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color="green", 
                  annotation_text="Good (80%)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                  annotation_text="Fair (60%)")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time/Frame",
        yaxis_title="Compliance Rate (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400
    )
    
    return fig


def create_violation_distribution_chart(violations: List[Dict]) -> go.Figure:
    """Create a pie chart showing violation type distribution
    
    Args:
        violations: List of violation dictionaries
        
    Returns:
        Plotly figure
    """
    if not violations:
        return go.Figure()
    
    # Count violation types
    violation_counts = {}
    for violation in violations:
        v_type = violation.get('type', 'Unknown')
        violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(violation_counts.keys()),
        values=list(violation_counts.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Violation Types Distribution",
        template="plotly_white",
        height=400
    )
    
    return fig


def save_detection_report(detections: List[Dict], compliance_stats: Dict, 
                         image_path: str = None) -> str:
    """Save detection report as JSON
    
    Args:
        detections: Detection results
        compliance_stats: Compliance statistics
        image_path: Path to processed image
        
    Returns:
        JSON report string
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_detections': len(detections),
            'total_people': compliance_stats.get('total_people', 0),
            'compliant_people': compliance_stats.get('compliant_people', 0),
            'violations': len(compliance_stats.get('violations', [])),
            'compliance_rate': compliance_stats.get('compliance_rate', 0.0)
        },
        'detections': detections,
        'compliance_stats': compliance_stats,
        'image_path': image_path
    }
    
    return json.dumps(report, indent=2)


def validate_image(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded image file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size too large (max 10MB)"
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
    if uploaded_file.type not in allowed_types:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_types)}"
    
    try:
        # Try to open image
        image = Image.open(uploaded_file)
        image.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def validate_video(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded video file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (max 200MB)
    if uploaded_file.size > 200 * 1024 * 1024:
        return False, "File size too large (max 200MB)"
    
    # Check file type
    allowed_types = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv']
    if uploaded_file.type not in allowed_types:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_types)}"
    
    return True, ""


def create_status_indicator(status: str, label: str = "") -> str:
    """Create a colored status indicator
    
    Args:
        status: Status type ('online', 'offline', 'warning', 'error')
        label: Optional label text
        
    Returns:
        HTML string for status indicator
    """
    colors = {
        'online': '#4CAF50',
        'offline': '#f44336',
        'warning': '#ff9800',
        'error': '#f44336',
        'processing': '#2196F3'
    }
    
    color = colors.get(status, '#757575')
    
    html = f"""
    <div style="display: inline-flex; align-items: center; margin: 5px 0;">
        <div style="
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            background-color: {color};
            margin-right: 8px;
            animation: {('pulse 2s infinite' if status == 'online' else 'none')};
        "></div>
        <span style="font-weight: 500;">{label}</span>
    </div>
    """
    
    return html


def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def calculate_processing_time(start_time: float, end_time: float) -> str:
    """Calculate and format processing time
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Formatted time duration
    """
    duration = end_time - start_time
    
    if duration < 60:
        return f"{duration:.1f} seconds"
    elif duration < 3600:
        minutes = duration // 60
        seconds = duration % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def create_metric_card(title: str, value: str, delta: str = None, 
                      delta_color: str = "normal") -> str:
    """Create a custom metric card
    
    Args:
        title: Metric title
        value: Metric value
        delta: Change value (optional)
        delta_color: Color for delta ('normal', 'inverse')
        
    Returns:
        HTML string for metric card
    """
    delta_html = ""
    if delta:
        delta_color_map = {
            'normal': '#28a745' if not delta.startswith('-') else '#dc3545',
            'inverse': '#dc3545' if not delta.startswith('-') else '#28a745'
        }
        color = delta_color_map.get(delta_color, '#6c757d')
        
        delta_html = f"""
        <div style="font-size: 0.8rem; color: {color}; margin-top: 4px;">
            {delta}
        </div>
        """
    
    html = f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    ">
        <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 4px;">
            {title}
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #212529;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    
    return html


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'memory_available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%"
        }
    except Exception:
        info = {'error': 'Could not retrieve system information'}
    
    return info


# CSS for animations and custom styling
CUSTOM_CSS = """
<style>
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}

.status-good {
    color: #28a745;
    font-weight: bold;
}

.status-warning {
    color: #ffc107;
    font-weight: bold;
}

.status-danger {
    color: #dc3545;
    font-weight: bold;
}

.detection-box {
    border: 2px solid #1f77b4;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    background: #f8f9fa;
}

.compliance-indicator {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
}

.compliance-good {
    background: #d4edda;
    color: #155724;
}

.compliance-warning {
    background: #fff3cd;
    color: #856404;
}

.compliance-danger {
    background: #f8d7da;
    color: #721c24;
}
</style>
"""
