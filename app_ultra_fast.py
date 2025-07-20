"""
PPE Compliance Monitoring System - Ultra Fast Version
Optimized for speed with cancellation support and enhanced UI
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import time
import io
import threading
from datetime import datetime, date, timedelta

# Import custom modules
from ppe_detection_engine import PPEDetectionEngine
from utils import validate_image, validate_video
from results_viewer import create_results_dashboard
from theme_manager import theme_manager

# Import attendance management
try:
    from attendance_manager import AttendanceManager
    ATTENDANCE_AVAILABLE = True
except ImportError:
    ATTENDANCE_AVAILABLE = False
    st.warning("⚠️ Attendance management not available. Please check installation.")

# Application Constants
PROGRESS_UPDATE_INTERVAL = 0.5  # seconds
RESULTS_HISTORY_LIMIT = 10  # maximum number of results to keep in history
MODEL_FILE_PATH = "best.pt"  # default model file path

# Try to import webcam component
try:
    from webcam_component import create_webcam_interface
    WEBCAM_AVAILABLE = True
except ImportError:
    WEBCAM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PPE Monitor Pro",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with enhanced animations and micro-interactions
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Main container with enhanced styling */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Global animations and transitions */
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Enhanced button hover effects */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 12px !important;
        font-weight: 600 !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Enhanced metric cards */
    .metric-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Enhanced loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    /* Apply animations to cards */
    .modern-action-card,
    .modern-upload-section,
    .result-panel {
        animation: slideInUp 0.6s ease-out;
    }

    /* Enhanced focus states for accessibility */
    .stButton > button:focus,
    .stSelectbox > div > div:focus,
    .stFileUploader > div:focus {
        outline: 2px solid var(--accent-color);
        outline-offset: 2px;
    }
    
    /* Ultra-modern header with theme support */
    .ultra-header {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--secondary-bg) 100%);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px var(--shadow);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        position: relative;
    }

    /* Help icon in corner */
    .help-icon {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: var(--accent-color);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px var(--shadow);
        transition: all 0.3s ease;
        font-size: 1.2rem;
        font-weight: bold;
        border: 2px solid var(--border-color);
    }

    .help-icon:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px var(--shadow);
        background: var(--secondary-bg);
    }


    
    .ultra-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px var(--shadow);
        color: var(--text-primary);
    }

    .ultra-header p {
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.95;
        color: var(--text-secondary);
    }

    /* Enhanced file uploader styling for main content */
    .stFileUploader {
        background: var(--card-bg) !important;
        border: 2px dashed var(--accent-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: var(--secondary-bg) !important;
        border: 2px dashed var(--accent-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 2rem !important;
        text-align: center !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* Speed metrics */
    .speed-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .speed-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    
    .speed-label {
        font-size: 0.8rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .status-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0 0 0.3rem 0;
        line-height: 1.2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .status-label {
        color: #5a6c7d;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.3;
    }

    /* Enhanced metric cards for better visibility */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 35px rgba(102, 126, 234, 0.4);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        line-height: 1.1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.3;
    }
    
    /* Progress styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Stop button styling */
    .stop-button {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
        min-height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
    }

    .stop-button:hover {
        background: linear-gradient(135deg, #ff3838 0%, #ff2f2f 100%) !important;
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 30px rgba(255, 71, 87, 0.6) !important;
    }

    .stop-button:active {
        transform: translateY(-1px) scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.5) !important;
    }

    .stop-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }

    .stop-button:hover::before {
        left: 100%;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9ff 0%, #e3e7ff 100%);
        text-align: center;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Enhanced text readability */
    .stMarkdown {
        line-height: 1.6;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #2c3e50;
        font-weight: 700;
        line-height: 1.3;
        margin-bottom: 1rem;
    }

    .stMarkdown p {
        color: #34495e;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }

    /* Improved dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Better metric display */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 3px solid #667eea;
    }

    .stMetric [data-testid="metric-container"] {
        background: transparent;
    }

    /* Enhanced info/success/error boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        line-height: 1.5;
    }

    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-left: 4px solid #28a745;
    }

    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        border-left: 4px solid #17a2b8;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border-left: 4px solid #dc3545;
    }

    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border-left: 4px solid #ffc107;
    }

    /* Responsive design improvements */
    @media (max-width: 768px) {
        .metric-card, .status-card {
            margin: 0.3rem 0;
            padding: 1rem;
        }

        .metric-value, .status-value {
            font-size: 1.8rem;
        }

        .metric-label, .status-label {
            font-size: 0.8rem;
        }

        .ultra-header h1 {
            font-size: 1.8rem;
        }

        .ultra-header p {
            font-size: 0.9rem;
        }
    }

    /* Improved scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_engine' not in st.session_state:
    st.session_state.detection_engine = None
    st.session_state.model_loaded = False
    st.session_state.processing = False
    st.session_state.stop_processing = None
    st.session_state.video_results = None
    st.session_state.processed_video_path = None

# Initialize attendance manager
if 'attendance_manager' not in st.session_state and ATTENDANCE_AVAILABLE:
    try:
        st.session_state.attendance_manager = AttendanceManager()
    except Exception as e:
        st.error(f"❌ Failed to initialize attendance system: {e}")
        st.session_state.attendance_manager = None
    st.session_state.processed_video_data = None
    st.session_state.show_results = False
    st.session_state.show_results_in_tabs = False
    st.session_state.current_video_path = None
    st.session_state.results_history = []

@st.cache_resource
def load_detection_engine():
    """Load and cache the detection engine"""
    try:
        engine = PPEDetectionEngine(MODEL_FILE_PATH)
        return engine, True
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        return None, False

def create_speed_sidebar():
    """Create speed-optimized sidebar"""
    # Get current theme for consistent styling
    current_theme = theme_manager.get_current_theme()
    theme_config = theme_manager.get_theme_config(current_theme)

    # Add animated logo at the top of sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin: 1rem 0 2rem 0;">
        <div class="animated-logo">
            <div class="logo-container">
                <div class="logo-icon">🦺</div>
                <div class="logo-text">PPE Monitor Pro</div>
                <div class="logo-pulse"></div>
            </div>
        </div>
    </div>

    <style>
    .animated-logo {{
        position: relative;
        display: inline-block;
        animation: logoFloat 3s ease-in-out infinite;
    }}

    .logo-container {{
        position: relative;
        background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%);
        border-radius: 20px;
        padding: 1.5rem 1rem;
        box-shadow: 0 8px 32px {theme_config['shadow']};
        border: 2px solid {theme_config['border_color']};
        backdrop-filter: blur(10px);
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .logo-container:hover {{
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 12px 40px {theme_config['shadow_hover']};
        background: linear-gradient(135deg, {theme_config['info_color']} 0%, {theme_config['accent_color']} 100%);
    }}

    .logo-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        animation: iconSpin 4s linear infinite;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }}

    .logo-text {{
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        animation: textGlow 2s ease-in-out infinite alternate;
    }}

    .logo-pulse {{
        position: absolute;
        top: 50%;
        left: 50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(-50%, -50%) scale(0);
        animation: pulse 2s ease-in-out infinite;
    }}

    @keyframes logoFloat {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        25% {{ transform: translateY(-5px) rotate(1deg); }}
        50% {{ transform: translateY(-3px) rotate(0deg); }}
        75% {{ transform: translateY(-7px) rotate(-1deg); }}
    }}

    @keyframes iconSpin {{
        0% {{ transform: rotate(0deg) scale(1); }}
        25% {{ transform: rotate(90deg) scale(1.1); }}
        50% {{ transform: rotate(180deg) scale(1); }}
        75% {{ transform: rotate(270deg) scale(1.1); }}
        100% {{ transform: rotate(360deg) scale(1); }}
    }}

    @keyframes textGlow {{
        0% {{ text-shadow: 0 2px 4px rgba(0,0,0,0.3), 0 0 10px rgba(255,255,255,0.1); }}
        100% {{ text-shadow: 0 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.3); }}
    }}

    @keyframes pulse {{
        0% {{ transform: translate(-50%, -50%) scale(0); opacity: 1; }}
        50% {{ transform: translate(-50%, -50%) scale(1); opacity: 0.5; }}
        100% {{ transform: translate(-50%, -50%) scale(1.5); opacity: 0; }}
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .logo-container {{
            padding: 1rem 0.8rem;
        }}
        .logo-icon {{
            font-size: 2rem;
        }}
        .logo-text {{
            font-size: 0.8rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Add custom styling for the help button
    help_button_style = f"""
    <style>
    div[data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%) !important;
        color: white !important;
        border: 2px solid {theme_config['border_color']} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 12px {theme_config['shadow']} !important;
    }}

    div[data-testid="stSidebar"] .stButton > button:hover {{
        background: linear-gradient(135deg, {theme_config['info_color']} 0%, {theme_config['accent_color']} 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {theme_config['shadow_hover']} !important;
    }}

    div[data-testid="stSidebar"] .stButton > button:active {{
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px {theme_config['shadow']} !important;
    }}
    </style>
    """
    st.markdown(help_button_style, unsafe_allow_html=True)

    # Help button in sidebar
    if st.sidebar.button("📖 Open Help & Status Guide",
                        use_container_width=True,
                        help="View system status, usage tips, and troubleshooting",
                        type="secondary"):
        st.session_state.show_help_modal = True

    st.sidebar.markdown("---")

    # Speed Settings header
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h3 style="margin: 0;">🚀 Speed Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status
    if st.session_state.model_loaded:
        st.sidebar.success("🟢 Model Ready")
    else:
        st.sidebar.error("🔴 Model Error")
    
    # Default speed settings (now controlled from main area if needed)
    # Set default to Ultra Fast for best performance
    settings = {"conf": 0.6, "iou": 0.5, "skip": 4, "res": 640}
    
    st.sidebar.markdown("---")
    
    # Set default camera quality settings (now controlled from main area)
    if 'camera_quality_selection' not in st.session_state:
        st.session_state.camera_quality_selection = "HD (720p)"

    # Default camera quality settings
    camera_quality_options = {
        "Standard (480p)": {"width": 640, "height": 480, "label": "Standard", "icon": "🟡"},
        "HD (720p)": {"width": 1280, "height": 720, "label": "HD", "icon": "🟢"},
        "Full HD (1080p)": {"width": 1920, "height": 1080, "label": "Full HD", "icon": "🔵"},
        "Ultra HD (1440p)": {"width": 2560, "height": 1440, "label": "Ultra HD", "icon": "🟣"}
    }

    # Apply default camera quality settings
    quality_config = camera_quality_options[st.session_state.camera_quality_selection]
    settings['camera_quality'] = quality_config['label']
    settings['camera_width'] = quality_config['width']
    settings['camera_height'] = quality_config['height']
    settings['camera_icon'] = quality_config['icon']

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced"):
        settings['conf'] = st.slider("Confidence", 0.1, 1.0, settings['conf'], 0.05)
        settings['iou'] = st.slider("IoU", 0.1, 1.0, settings['iou'], 0.05)
        settings['skip'] = st.slider("Skip Frames", 1, 5, settings['skip'])
        settings['res'] = st.selectbox("Resolution", [640, 1280],
                                     index=0 if settings['res'] == 640 else 1)

        st.markdown("**🎬 Video Output Quality**")
        settings['smooth_detections'] = st.checkbox(
            "Smooth Detection Overlay",
            value=True,
            help="Prevents flickering by maintaining detections across skipped frames"
        )

        if settings['smooth_detections']:
            st.info("✨ Smooth overlay prevents bounding box flickering in downloaded videos")
        else:
            st.warning("⚠️ Detections may flicker in downloaded video when speed boost is used")

    # Face Recognition settings
    with st.sidebar.expander("👤 Face Recognition"):
        face_engine = st.session_state.detection_engine.get_face_engine()

        if face_engine:
            st.success("🟢 Face Recognition Ready")

            # Dataset info
            dataset_info = face_engine.get_dataset_info()
            if dataset_info['total_people'] > 0:
                st.info(f"📊 {dataset_info['total_people']} people trained with {dataset_info['total_samples']} samples")

                # Show recognized people
                for person in dataset_info['people']:
                    st.text(f"• {person['name']}: {person['samples']} samples")
            else:
                st.warning("⚠️ No face data available. Train the model first.")

            # Confidence threshold
            if face_engine.is_trained:
                new_threshold = st.slider("Recognition Threshold", 0, 100,
                                        face_engine.confidence_threshold, 5,
                                        help="Higher values = more strict recognition")
                if new_threshold != face_engine.confidence_threshold:
                    face_engine.update_confidence_threshold(new_threshold)
                    st.success(f"Threshold updated to {new_threshold}%")
        else:
            st.error("🔴 Face Recognition Unavailable")


    return settings



def process_video_with_cancellation(video_path, output_path, settings):
    """Process video with cancellation support and persistent results"""

    # Create stop flag
    stop_flag = threading.Event()
    st.session_state.stop_processing = stop_flag

    # Enhanced progress containers with modern styling
    current_theme = theme_manager.get_current_theme()
    theme_config = theme_manager.get_theme_config(current_theme)

    progress_container = st.container()

    with progress_container:

        # Enhanced progress layout with better positioned stop button
        st.markdown(f"""
        <style>
        /* Enhanced progress bar styling */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%) !important;
            background-size: 200% 100% !important;
            animation: progressFlow 2s ease-in-out infinite !important;
            border-radius: 10px !important;
            height: 14px !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }}

        .stProgress > div > div > div {{
            background: {theme_config['border_color']} !important;
            border-radius: 10px !important;
            height: 14px !important;
            overflow: hidden !important;
        }}

        @keyframes progressFlow {{
            0% {{
                background-position: 200% 0;
            }}
            100% {{
                background-position: -200% 0;
            }}
        }}

        /* Modern stop button with symbol only */
        div[data-testid="stButton"] > button[key="stop_btn"] {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            padding: 0.5rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3), 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            min-height: 40px !important;
            min-width: 40px !important;
            width: 40px !important;
            height: 40px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            margin: 0.4rem 0 !important;
            line-height: 1 !important;
            position: relative !important;
            overflow: hidden !important;
        }}

        div[data-testid="stButton"] > button[key="stop_btn"]::before {{
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
            transition: left 0.5s ease !important;
        }}

        div[data-testid="stButton"] > button[key="stop_btn"]:hover {{
            background: linear-gradient(135deg, #ff5252 0%, #e53935 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4), 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        }}

        div[data-testid="stButton"] > button[key="stop_btn"]:hover::before {{
            left: 100% !important;
        }}

        div[data-testid="stButton"] > button[key="stop_btn"]:active {{
            background: linear-gradient(135deg, #e53935 0%, #d32f2f 100%) !important;
            transform: translateY(0) !important;
            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }}

        /* Simple stop button container */
        .stop-button-container {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-top: 0.5rem !important;
        }}

        /* Responsive design for symbol button */
        @media (max-width: 768px) {{
            div[data-testid="stButton"] > button[key="stop_btn"] {{
                font-size: 1.1rem !important;
                padding: 0.4rem !important;
                min-height: 38px !important;
                min-width: 38px !important;
                width: 38px !important;
                height: 38px !important;
            }}
        }}

        @media (max-width: 480px) {{
            div[data-testid="stButton"] > button[key="stop_btn"] {{
                font-size: 1rem !important;
                padding: 0.35rem !important;
                min-height: 36px !important;
                min-width: 36px !important;
                width: 36px !important;
                height: 36px !important;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Modern stop button with symbol only
        st.markdown('<div class="stop-button-container">', unsafe_allow_html=True)
        _, col_center, _ = st.columns([1, 1, 1])
        with col_center:
            if st.button("⏹", key="stop_btn"):
                stop_flag.set()
                st.warning("⏹️ Stopping processing...")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Clean processing function with simple status updates
    def update_progress(progress):
        if not stop_flag.is_set():
            progress_bar.progress(progress / 100)
            # Simple status text without animations
            status_text.markdown(f"🎬 Processing: {progress:.0f}%")



    start_time = time.time()

    try:
        # Process video with configurable detection smoothing
        results = st.session_state.detection_engine.process_video(
            video_path,
            output_path,
            settings['conf'],
            settings['iou'],
            update_progress,
            settings['skip'],
            settings['res'],
            stop_flag,
            persistent_overlay=settings.get('smooth_detections', True)
        )

        processing_time = time.time() - start_time

        if stop_flag.is_set() or results.get('cancelled'):
            st.warning("⏹️ Processing cancelled by user")
            return None

        if 'error' not in results:
            # Read and store the processed video data
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    video_data = f.read()
                st.session_state.processed_video_data = video_data

            # Store results in session state for persistence
            st.session_state.video_results = results
            st.session_state.processed_video_path = output_path
            st.session_state.show_results = True

            # Add to history
            # Ensure datetime is available (defensive import)
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except NameError:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            history_entry = {
                'timestamp': timestamp,
                'results': results,
                'video_data': st.session_state.processed_video_data,
                'processing_time': processing_time,
                'settings': settings
            }
            st.session_state.results_history.append(history_entry)

            # Keep only last N results in history
            if len(st.session_state.results_history) > RESULTS_HISTORY_LIMIT:
                st.session_state.results_history = st.session_state.results_history[-RESULTS_HISTORY_LIMIT:]

            # Display comprehensive results using the results viewer
            create_results_dashboard(results, output_path, processing_time, settings, unique_id="main")

            return results
        else:
            st.error(f"❌ Processing failed: {results['error']}")
            return None

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

    finally:
        st.session_state.processing = False
        st.session_state.stop_processing = None

def main():
    """Main application"""

    # Initialize and apply theme
    theme_manager.apply_theme_css()

    # Add theme toggle to sidebar with enhanced styling
    with st.sidebar:
        st.markdown("---")

        # Theme settings header with proper styling
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        st.markdown(f"""
        <h3 style="color: {theme_config['text_primary']} !important; font-weight: 600; margin-bottom: 1rem;">
            🎨 Theme Settings
        </h3>
        """, unsafe_allow_html=True)

        theme_manager.create_theme_toggle()

        # Show current theme indicator with proper theming
        st.markdown(f"""
        <div style="
            background: {theme_config['accent_color']};
            color: white;
            padding: 0.5rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
            font-weight: 600;
            border: 1px solid {theme_config['border_color']};
        ">
            {theme_config['icon']} {theme_config['name']} Active
        </div>
        """, unsafe_allow_html=True)



    # Enhanced ultra-modern header with status
    webcam_status = "🟢 Available" if WEBCAM_AVAILABLE else "🔴 Unavailable"
    model_status = "🟢 Ready" if st.session_state.model_loaded else "🔴 Error"

    st.markdown(f"""
    <div class="ultra-header">
        <h1>🦺 PPE Monitor Pro</h1>
        <p>Ultra-fast AI workplace safety monitoring with real-time analytics</p>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
            <span style="margin-right: 2rem;">🤖 AI Model: {model_status}</span>
            <span style="margin-right: 2rem;">📹 Live Detection: {webcam_status}</span>
            <span>⚡ Processing: Ultra-Fast Mode</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


    
    # Load detection engine
    if st.session_state.detection_engine is None:
        with st.spinner("🔄 Loading AI model..."):
            engine, loaded = load_detection_engine()
            st.session_state.detection_engine = engine
            st.session_state.model_loaded = loaded
    
    if not st.session_state.model_loaded:
        st.error("❌ AI model not available. Please check your model file.")
        st.stop()
    
    # Speed sidebar
    settings = create_speed_sidebar()

    # Initialize help modal state
    if 'show_help_modal' not in st.session_state:
        st.session_state.show_help_modal = False

    # Help modal using Streamlit dialog
    if st.session_state.show_help_modal:
        # Create modal using Streamlit's dialog
        @st.dialog("ℹ️ Help & Status Guide")
        def show_help_dialog():
            # Getting Started section
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            - **Live Detection:** Click START to begin camera detection
            - **Video Processing:** Upload MP4/AVI files for analysis
            - **Image Analysis:** Upload JPG/PNG for instant detection
            - **Results History:** View and download previous analyses
            """)

            st.markdown("---")

            # Analytics Timeline section
            st.markdown("### 📊 Analytics Timeline")
            st.markdown("""
            - **0-3 frames:** 🔄 Initializing detection engine
            - **3-10 frames:** 📊 Building analytics report
            - **10+ frames:** ✅ Full analytics available
            - **30+ frames:** 🎯 High quality analytics & insights
            """)

            st.markdown("---")

            # Detection Colors section
            st.markdown("### 🎨 Detection Colors")
            st.markdown("""
            - 🟢 **Green:** PPE present (compliant)
            - 🔴 **Red:** PPE missing (violation)
            - 🟡 **Yellow:** Person detected
            - 🟠 **Orange:** Safety equipment
            - 🟣 **Purple:** Machinery/vehicles
            """)

            st.markdown("---")

            # Control Functions section
            st.markdown("### 🎛️ Control Functions")
            st.markdown("""
            - **🔄 Reset:** Complete system reset
            - **📊 Refresh:** Force immediate display update
            - **⏸️ Pause/▶️ Resume:** Pause/resume analytics
            - **💾 Export:** Download session statistics
            - **🚀 Process:** Start video/image analysis
            """)

            st.markdown("---")

            # Performance Info section
            st.markdown("### ⏱️ Performance Info")
            st.markdown("""
            - **Basic stats:** Available immediately
            - **Analytics report:** Ready in ~3-5 seconds
            - **High quality data:** Achieved after ~30 seconds
            - **Ultra-fast mode:** Up to 8x processing speed
            """)

            st.markdown("---")

            # System Status section
            st.markdown("### 🔧 System Status")
            st.markdown(f"""
            - **AI Model:** {model_status}
            - **Live Detection:** {webcam_status}
            - **Processing Mode:** Ultra-Fast
            - **Version:** PPE Monitor Pro v2.1
            """)

            st.markdown("---")

            # Close button
            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button("✕ Close", type="primary", use_container_width=True):
                    st.session_state.show_help_modal = False
                    st.rerun()

        show_help_dialog()

    # Main interface
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📹 Live Detection", "🎥 Ultra-Fast Video", "📷 Instant Analysis", "📋 Results History", "📊 Export Session Data", "👤 Face Recognition", "📊 Live Attendance"])
    
    with tab1:
        if WEBCAM_AVAILABLE:
            # Import the analytics function
            from webcam_component import create_webcam_analytics

            # Create the main webcam interface
            webrtc_ctx = create_webcam_interface(st.session_state.detection_engine, settings)

            # Add analytics section below the webcam interface
            st.markdown("---")

            # Enhanced analytics with better error handling and status display
            if webrtc_ctx and hasattr(st.session_state, 'webcam_detector'):
                try:
                    # Check if camera is active
                    camera_active = webrtc_ctx.state.playing

                    # Track camera state changes and manage session accordingly
                    if 'previous_camera_state_main' not in st.session_state:
                        st.session_state.previous_camera_state_main = False

                    # Detect camera state changes
                    if camera_active != st.session_state.previous_camera_state_main:
                        if camera_active:
                            # Camera just started
                            st.session_state.webcam_detector.start_session()
                        else:
                            # Camera just stopped
                            st.session_state.webcam_detector.stop_session()
                        st.session_state.previous_camera_state_main = camera_active

                    if camera_active:
                        # Get latest stats with error handling
                        try:
                            latest_stats = st.session_state.webcam_detector.get_latest_stats()
                        except Exception as e:
                            st.error(f"Error retrieving analytics: {str(e)}")
                            latest_stats = None

                        if latest_stats and latest_stats['frame_count'] > 0:
                            # Enhanced analytics readiness status
                            frame_count = latest_stats['frame_count']
                            if frame_count < 3:
                                st.info(f"🔄 **Initializing Analytics...** ({frame_count}/3 frames)")
                                progress = frame_count / 3
                                st.progress(progress, text=f"Processing frame {frame_count}")
                            elif frame_count < 10:
                                st.warning(f"📊 **Building Report...** ({frame_count}/10 frames for full analytics)")
                                progress = frame_count / 10
                                st.progress(progress, text=f"Analytics quality: {progress*100:.0f}%")
                            else:
                                st.success("✅ **Full Analytics Available**")

                        # Show the comprehensive analytics
                        create_webcam_analytics(st.session_state.webcam_detector)

                    else:
                        st.info("📹 **Start the camera to see live analytics**")

                        # Enhanced last session data display
                        try:
                            latest_stats = st.session_state.webcam_detector.get_latest_stats()
                            if latest_stats and latest_stats['frame_count'] > 0:
                                st.markdown("---")
                                st.markdown("### 📊 **Last Session Summary**")

                                # Enhanced metrics display
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Frames Processed", f"{latest_stats['frame_count']:,}")
                                with col2:
                                    duration = latest_stats.get('session_duration', 0)
                                    minutes = int(duration // 60)
                                    seconds = int(duration % 60)
                                    st.metric("Duration", f"{minutes:02d}:{seconds:02d}")
                                with col3:
                                    st.metric("Final Compliance", f"{latest_stats['compliance_rate']:.0f}%")
                                with col4:
                                    total_people = latest_stats.get('total_people_session', 0)
                                    st.metric("Total People", f"{total_people:,}")

                                # Additional session info
                                if latest_stats.get('avg_compliance_rate', 0) > 0:
                                    avg_compliance = latest_stats['avg_compliance_rate']
                                    if avg_compliance >= 90:
                                        st.success(f"🟢 **Excellent Session** - {avg_compliance:.1f}% average compliance")
                                    elif avg_compliance >= 75:
                                        st.info(f"✅ **Good Session** - {avg_compliance:.1f}% average compliance")
                                    elif avg_compliance >= 50:
                                        st.warning(f"⚠️ **Fair Session** - {avg_compliance:.1f}% average compliance")
                                    else:
                                        st.error(f"❌ **Poor Session** - {avg_compliance:.1f}% average compliance")
                        except Exception as e:
                            st.warning(f"Could not retrieve last session data: {str(e)}")

                except Exception as e:
                    st.error(f"Analytics system error: {str(e)}")

            else:
                st.info("📊 **Analytics will appear here once you start the camera**")

                # Enhanced preview of analytics features
                with st.expander("📈 **Preview: Available Analytics Features**", expanded=False):
                    col_preview1, col_preview2 = st.columns(2)

                    with col_preview1:
                        st.markdown("""
                        **🔴 Real-time Monitoring:**
                        - 👥 People detection count and trends
                        - ⚠️ PPE violation tracking and alerts
                        - 📊 Live compliance rate calculation
                        - 🎥 Processing performance metrics

                        **📈 Session Statistics:**
                        - ⏱️ Session duration and frame count
                        - 📊 Average compliance over time
                        - 🎯 Detection consistency analysis
                        - ⚡ Real-time processing performance
                        """)

                    with col_preview2:
                        st.markdown("""
                        **🎯 Performance Indicators:**
                        - ✅ Detection accuracy rates
                        - 🔄 Frame processing speed (FPS)
                        - 📊 Compliance trend analysis
                        - 🎯 System performance metrics

                        **💾 Export Capabilities:**
                        - 📄 JSON reports with complete data
                        - 📊 CSV files for spreadsheet analysis
                        - 📋 Executive summary reports
                        - 📈 Timeline data and trends
                        """)
        else:
            st.warning("⚠️ **Webcam features not available**")
            st.info("Install streamlit-webrtc to enable live detection:")
            st.code("pip install streamlit-webrtc", language="bash")

            # Show what features would be available
            st.markdown("### 🔮 Available Features (After Installation)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🎥 Live Detection:**
                - Real-time PPE compliance monitoring
                - Webcam integration with detection overlay
                - Live violation alerts and notifications
                - Adjustable detection sensitivity
                """)

            with col2:
                st.markdown("""
                **📊 Live Analytics:**
                - Real-time compliance statistics
                - People counting and tracking
                - Session performance metrics
                - Exportable detection reports
                """)
    
    with tab2:
        # Get current theme for consistent styling
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        # Modern header with glassmorphism effect
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['info_color']}10 100%);
            backdrop-filter: blur(10px);
            border: 1px solid {theme_config['border_color']};
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px {theme_config['shadow']};
        ">
            <h2 style="
                color: {theme_config['text_primary']};
                margin: 0;
                font-weight: 700;
                font-size: 2rem;
                background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">
                🚀 Ultra-Fast Video Processing
            </h2>
            <p style="
                color: {theme_config['text_secondary']};
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
                font-weight: 500;
            ">
                Advanced AI-powered PPE compliance analysis with lightning-fast processing
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced drag & drop section with modern glassmorphism design
        st.markdown(f"""
        <style>
        /* Modern upload container with glassmorphism */
        .modern-upload-container {{
            position: relative;
            margin: 2rem 0;
        }}

        .modern-upload-section {{
            border: 2px dashed {theme_config['accent_color']}80;
            border-radius: 24px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, {theme_config['card_bg']}90 0%, {theme_config['secondary_bg']}70 100%);
            backdrop-filter: blur(20px);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px {theme_config['shadow']};
        }}

        .modern-upload-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, {theme_config['accent_color']}20 50%, transparent 70%);
            transform: translateX(-100%);
            transition: transform 0.8s ease;
        }}

        .modern-upload-section:hover {{
            border-color: {theme_config['accent_color']};
            background: linear-gradient(135deg, {theme_config['card_bg']}95 0%, {theme_config['secondary_bg']}80 100%);
            transform: translateY(-4px);
            box-shadow: 0 16px 48px {theme_config['shadow_hover']};
        }}

        .modern-upload-section:hover::before {{
            transform: translateX(100%);
        }}

        .upload-icon-modern {{
            font-size: 4rem;
            margin-bottom: 1.5rem;
            opacity: 0.8;
            transition: all 0.4s ease;
            background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .modern-upload-section:hover .upload-icon-modern {{
            opacity: 1;
            transform: scale(1.1) rotate(5deg);
        }}

        .upload-text-modern {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {theme_config['text_primary']};
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px {theme_config['shadow']};
        }}

        .upload-subtitle-modern {{
            font-size: 1.1rem;
            color: {theme_config['text_secondary']};
            margin-bottom: 1.5rem;
            font-weight: 500;
            line-height: 1.6;
        }}

        .supported-formats-modern {{
            font-size: 0.9rem;
            color: {theme_config['text_muted']};
            margin-top: 1rem;
            padding: 0.8rem 1.5rem;
            background: {theme_config['tertiary_bg']}80;
            backdrop-filter: blur(10px);
            border-radius: 25px;
            display: inline-block;
            font-weight: 500;
            border: 1px solid {theme_config['border_color']};
        }}

        /* Enhanced file uploader styling */
        .stFileUploader {{
            margin-top: -1rem;
        }}

        .stFileUploader > div {{
            border: none !important;
            background: transparent !important;
        }}

        .stFileUploader > div > div {{
            border: 2px dashed {theme_config['accent_color']}80 !important;
            border-radius: 24px !important;
            background: linear-gradient(135deg, {theme_config['card_bg']}90 0%, {theme_config['secondary_bg']}70 100%) !important;
            backdrop-filter: blur(20px) !important;
            padding: 3rem 2rem !important;
            text-align: center !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px {theme_config['shadow']} !important;
            min-height: 180px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }}

        .stFileUploader > div > div:hover {{
            border-color: {theme_config['accent_color']} !important;
            background: linear-gradient(135deg, {theme_config['card_bg']}95 0%, {theme_config['secondary_bg']}80 100%) !important;
            transform: translateY(-4px) !important;
            box-shadow: 0 16px 48px {theme_config['shadow_hover']} !important;
        }}

        /* Drag states with enhanced visual feedback */
        .stFileUploader > div > div[data-testid="stFileUploaderDropzone"] {{
            border: 2px dashed {theme_config['accent_color']}80 !important;
            border-radius: 24px !important;
            background: linear-gradient(135deg, {theme_config['card_bg']}90 0%, {theme_config['secondary_bg']}70 100%) !important;
            backdrop-filter: blur(20px) !important;
            padding: 3rem 2rem !important;
            text-align: center !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            min-height: 180px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex-direction: column !important;
            box-shadow: 0 8px 32px {theme_config['shadow']} !important;
        }}

        .stFileUploader > div > div[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {theme_config['accent_color']} !important;
            background: linear-gradient(135deg, {theme_config['card_bg']}95 0%, {theme_config['secondary_bg']}80 100%) !important;
            transform: translateY(-4px) !important;
            box-shadow: 0 16px 48px {theme_config['shadow_hover']} !important;
        }}

        /* Active drag state */
        .stFileUploader > div > div[data-testid="stFileUploaderDropzone"].st-emotion-cache-1kyxreq {{
            border-color: {theme_config['success_color']} !important;
            background: linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['card_bg']}90 100%) !important;
            box-shadow: 0 16px 48px {theme_config['success_color']}30 !important;
            transform: scale(1.02) !important;
        }}

        /* File uploader text styling */
        .stFileUploader label, .stFileUploader div, .stFileUploader span {{
            color: {theme_config['text_primary']} !important;
            font-weight: 500 !important;
        }}
        </style>

        <script>
        // Enhanced drag and drop with smooth animations
        document.addEventListener('DOMContentLoaded', function() {{
            const fileUploaders = document.querySelectorAll('[data-testid="stFileUploaderDropzone"]');

            fileUploaders.forEach(uploader => {{
                uploader.addEventListener('dragenter', function(e) {{
                    e.preventDefault();
                    this.style.borderColor = '{theme_config['success_color']}';
                    this.style.background = 'linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['card_bg']}90 100%)';
                    this.style.transform = 'scale(1.02)';
                    this.style.boxShadow = '0 16px 48px {theme_config['success_color']}30';
                }});

                uploader.addEventListener('dragleave', function(e) {{
                    e.preventDefault();
                    this.style.borderColor = '{theme_config['accent_color']}80';
                    this.style.background = 'linear-gradient(135deg, {theme_config['card_bg']}90 0%, {theme_config['secondary_bg']}70 100%)';
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = '0 8px 32px {theme_config['shadow']}';
                }});

                uploader.addEventListener('drop', function(e) {{
                    this.style.borderColor = '{theme_config['accent_color']}80';
                    this.style.background = 'linear-gradient(135deg, {theme_config['card_bg']}90 0%, {theme_config['secondary_bg']}70 100%)';
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = '0 8px 32px {theme_config['shadow']}';
                }});
            }});
        }});
        </script>
        """, unsafe_allow_html=True)

        # Modern file uploader with enhanced styling
        uploaded_video = st.file_uploader(
            "🎬 **Drag and drop your video here** or click to browse",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your video file for ultra-fast PPE compliance analysis • Supported: MP4, AVI, MOV, MKV • Max 200MB",
            label_visibility="collapsed"
        )

        # Show modern upload tips when no file is selected
        if not uploaded_video:
            st.markdown(f"""
            <div style="
                margin-top: 1.5rem;
                padding: 2rem;
                background: linear-gradient(135deg, {theme_config['accent_color']}08 0%, {theme_config['info_color']}05 100%);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                border: 1px solid {theme_config['border_color']};
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        font-size: 1.5rem;
                        margin-right: 0.8rem;
                    ">💡</div>
                    <h4 style="
                        color: {theme_config['text_primary']};
                        margin: 0;
                        font-weight: 700;
                        font-size: 1.3rem;
                    ">Upload Guidelines for Best Results</h4>
                </div>
                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                ">
                    <div style="
                        background: {theme_config['card_bg']}80;
                        backdrop-filter: blur(10px);
                        padding: 1.2rem;
                        border-radius: 12px;
                        border: 1px solid {theme_config['border_color']};
                    ">
                        <div style="color: {theme_config['success_color']}; font-size: 1.2rem; margin-bottom: 0.5rem;">🎯</div>
                        <strong style="color: {theme_config['text_primary']};">Best Quality:</strong>
                        <p style="color: {theme_config['text_secondary']}; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                            Use 1080p or higher resolution videos for optimal detection accuracy
                        </p>
                    </div>
                    <div style="
                        background: {theme_config['card_bg']}80;
                        backdrop-filter: blur(10px);
                        padding: 1.2rem;
                        border-radius: 12px;
                        border: 1px solid {theme_config['border_color']};
                    ">
                        <div style="color: {theme_config['warning_color']}; font-size: 1.2rem; margin-bottom: 0.5rem;">💡</div>
                        <strong style="color: {theme_config['text_primary']};">Good Lighting:</strong>
                        <p style="color: {theme_config['text_secondary']}; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                            Ensure people and PPE equipment are clearly visible
                        </p>
                    </div>
                    <div style="
                        background: {theme_config['card_bg']}80;
                        backdrop-filter: blur(10px);
                        padding: 1.2rem;
                        border-radius: 12px;
                        border: 1px solid {theme_config['border_color']};
                    ">
                        <div style="color: {theme_config['info_color']}; font-size: 1.2rem; margin-bottom: 0.5rem;">📹</div>
                        <strong style="color: {theme_config['text_primary']};">Stable Footage:</strong>
                        <p style="color: {theme_config['text_secondary']}; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                            Avoid shaky or blurry videos for better analysis
                        </p>
                    </div>
                    <div style="
                        background: {theme_config['card_bg']}80;
                        backdrop-filter: blur(10px);
                        padding: 1.2rem;
                        border-radius: 12px;
                        border: 1px solid {theme_config['border_color']};
                    ">
                        <div style="color: {theme_config['accent_color']}; font-size: 1.2rem; margin-bottom: 0.5rem;">🔄</div>
                        <strong style="color: {theme_config['text_primary']};">Multiple Angles:</strong>
                        <p style="color: {theme_config['text_secondary']}; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                            Include different viewpoints for comprehensive analysis
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_video:
            # Validate file
            is_valid, error = validate_video(uploaded_video)
            if not is_valid:
                st.error(f"❌ {error}")
                st.stop()

            # File info with modern success message
            file_size = uploaded_video.size / (1024 * 1024)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['success_color']}05 100%);
                border: 1px solid {theme_config['success_color']}40;
                border-radius: 16px;
                padding: 1.2rem;
                margin: 1.5rem 0;
                display: flex;
                align-items: center;
                gap: 1rem;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="
                    background: {theme_config['success_color']};
                    color: white;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.2rem;
                    font-weight: bold;
                ">✓</div>
                <div>
                    <h4 style="color: {theme_config['text_primary']}; margin: 0; font-weight: 600;">
                        Video uploaded successfully!
                    </h4>
                    <p style="color: {theme_config['text_secondary']}; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                        📁 {uploaded_video.name} • 📊 {file_size:.1f} MB • 🚀 Ready for processing
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Add a small delay to show the success message
            time.sleep(0.1)

            # Modern two-column layout
            col1, col2 = st.columns([2.2, 1], gap="large")

            with col1:
                # Video preview with modern styling
                st.markdown(f"""
                <div style="
                    background: {theme_config['card_bg']};
                    border-radius: 20px;
                    padding: 1rem;
                    border: 1px solid {theme_config['border_color']};
                    box-shadow: 0 8px 32px {theme_config['shadow']};
                    backdrop-filter: blur(10px);
                ">
                    <h4 style="
                        color: {theme_config['text_primary']};
                        margin: 0 0 1rem 0;
                        font-weight: 600;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    ">
                        📹 Video Preview
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                st.video(uploaded_video)

            with col2:

                # Speed info display with modern design
                speed_boost = settings['skip']
                estimated_speed = f"{speed_boost}x faster"

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['info_color']}10 100%);
                    border: 1px solid {theme_config['accent_color']}40;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 4px 20px {theme_config['shadow']};
                ">
                    <div style="
                        font-size: 2.5rem;
                        font-weight: 800;
                        background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        margin-bottom: 0.5rem;
                    ">{estimated_speed}</div>
                    <div style="
                        color: {theme_config['text_secondary']};
                        font-weight: 600;
                        font-size: 1rem;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">Processing Speed</div>
                    <div style="
                        color: {theme_config['text_muted']};
                        font-size: 0.85rem;
                        margin-top: 0.5rem;
                    ">Optimized for ultra-fast analysis</div>
                </div>
                """, unsafe_allow_html=True)

                # Modern process button
                if not st.session_state.processing:
                    # Custom styled button
                    st.markdown(f"""
                    <style>
                    .process-button {{
                        background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%);
                        color: white;
                        border: none;
                        border-radius: 16px;
                        padding: 1rem 2rem;
                        font-size: 1.1rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        cursor: pointer;
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        box-shadow: 0 8px 32px {theme_config['accent_color']}40;
                        width: 100%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 0.8rem;
                        position: relative;
                        overflow: hidden;
                    }}

                    .process-button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 12px 40px {theme_config['accent_color']}60;
                    }}

                    .process-button::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: -100%;
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                        transition: left 0.5s;
                    }}

                    .process-button:hover::before {{
                        left: 100%;
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                    if st.button("🚀 PROCESS VIDEO", type="primary", use_container_width=True, key="process_video_btn"):
                        st.session_state.processing = True

                        # Save uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(uploaded_video.read())
                            video_path = tmp.name

                        output_path = video_path.replace('.mp4', '_processed.mp4')

                        # Process with cancellation
                        results = process_video_with_cancellation(video_path, output_path, settings)

                        # Results are now displayed by the processing function
                        # and stored in session state for persistence

                        # Cleanup
                        for path in [video_path, output_path]:
                            if os.path.exists(path):
                                try:
                                    os.unlink(path)
                                except:
                                    pass

                        st.session_state.processing = False
                        st.rerun()

                else:
                    # Processing indicator with modern design
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['warning_color']}15 0%, {theme_config['warning_color']}05 100%);
                        border: 1px solid {theme_config['warning_color']}40;
                        border-radius: 16px;
                        padding: 1.5rem;
                        text-align: center;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 4px 20px {theme_config['shadow']};
                    ">
                        <div style="
                            font-size: 2rem;
                            margin-bottom: 0.8rem;
                            animation: spin 2s linear infinite;
                        ">🔄</div>
                        <div style="
                            color: {theme_config['text_primary']};
                            font-weight: 600;
                            font-size: 1.1rem;
                            margin-bottom: 0.5rem;
                        ">Processing in Progress</div>
                        <div style="
                            color: {theme_config['text_secondary']};
                            font-size: 0.9rem;
                        ">Please wait while we analyze your video...</div>
                    </div>

                    <style>
                    @keyframes spin {{
                        from {{ transform: rotate(0deg); }}
                        to {{ transform: rotate(360deg); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)

        # Show previous results if available with modern layout
        if st.session_state.show_results and st.session_state.video_results:


            # Modern action buttons with enhanced styling
            st.markdown(f"""
            <style>
            .modern-action-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }}

            .modern-action-card {{
                background: {theme_config['card_bg']};
                border: 1px solid {theme_config['border_color']};
                border-radius: 20px;
                padding: 1.5rem;
                text-align: center;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 20px {theme_config['shadow']};
                backdrop-filter: blur(10px);
                position: relative;
                overflow: hidden;
            }}

            .modern-action-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 12px 40px {theme_config['shadow_hover']};
            }}

            .action-icon {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
                display: block;
            }}

            .action-title {{
                color: {theme_config['text_primary']};
                font-weight: 700;
                font-size: 1.1rem;
                margin-bottom: 0.8rem;
            }}

            .action-description {{
                color: {theme_config['text_secondary']};
                font-size: 0.9rem;
                margin-bottom: 1.5rem;
                line-height: 1.5;
            }}

            @media (max-width: 768px) {{
                .modern-action-grid {{
                    grid-template-columns: 1fr;
                    gap: 1rem;
                }}
                .modern-action-card {{
                    padding: 1.2rem;
                }}
            }}
            </style>
            """, unsafe_allow_html=True)

            col_btn1, col_btn2, col_btn3 = st.columns(3, gap="large")

            with col_btn1:
                st.markdown(f"""
                <div class="modern-action-card">
                    <div class="action-icon" style="color: {theme_config['accent_color']};">🔄</div>
                    <div class="action-title">Process New Video</div>
                    <div class="action-description">Clear current results and start fresh analysis</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔄 Process New Video",
                           use_container_width=True,
                           type="primary",
                           help="Clear current results and process a new video",
                           key="new_video_btn"):
                    # Clear previous results
                    st.session_state.show_results = False
                    st.session_state.show_results_in_tabs = False
                    st.session_state.video_results = None
                    st.session_state.processed_video_path = None
                    st.session_state.current_video_path = None

                    # Show success message with modern styling
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['success_color']}05 100%);
                        border: 1px solid {theme_config['success_color']}40;
                        border-radius: 12px;
                        padding: 1rem;
                        margin: 1rem 0;
                        text-align: center;
                        color: {theme_config['text_primary']};
                        font-weight: 600;
                    ">
                        ✅ Ready for new video processing!
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.5)
                    st.rerun()

            with col_btn2:
                st.markdown(f"""
                <div class="modern-action-card">
                    <div class="action-icon" style="color: {theme_config['info_color']};">👁️</div>
                    <div class="action-title">View Results Again</div>
                    <div class="action-description">Redisplay the comprehensive analysis results</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("👁️ View Results Again",
                           use_container_width=True,
                           type="secondary",
                           help="Redisplay the current analysis results",
                           key="view_results_btn"):
                    # Show results again using comprehensive viewer with improved file handling
                    if st.session_state.video_results:
                        video_path = st.session_state.processed_video_path

                        # Check if video file still exists, if not recreate it
                        if not video_path or not os.path.exists(video_path):
                            # Try to recreate temporary file from session data
                            if hasattr(st.session_state, 'processed_video_data') and st.session_state.processed_video_data:
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                        tmp_file.write(st.session_state.processed_video_data)
                                        video_path = tmp_file.name
                                    st.session_state.processed_video_path = video_path
                                    st.info("🔄 Video file recreated from session data")
                                except Exception as e:
                                    st.error(f"❌ Could not recreate video file: {str(e)}")
                                    video_path = None
                            else:
                                st.warning("⚠️ Video data no longer available in session. Showing results without video.")
                                video_path = None

                        # Set flag to show results in broader tabs below
                        st.session_state.show_results_in_tabs = True
                        st.session_state.current_video_path = video_path
                    else:
                        st.error("❌ Previous results no longer available")

            with col_btn3:
                st.markdown(f"""
                <div class="modern-action-card">
                    <div class="action-icon" style="color: {theme_config['danger_color']};">🗑️</div>
                    <div class="action-title">Clear Results</div>
                    <div class="action-description">Remove current results from memory</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🗑️ Clear Results",
                           use_container_width=True,
                           help="Remove current results from memory",
                           key="clear_results_btn"):
                    # Clear results
                    st.session_state.show_results = False
                    st.session_state.show_results_in_tabs = False
                    st.session_state.video_results = None
                    st.session_state.processed_video_path = None
                    st.session_state.current_video_path = None

                    # Show success message with modern styling
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['success_color']}05 100%);
                        border: 1px solid {theme_config['success_color']}40;
                        border-radius: 12px;
                        padding: 1rem;
                        margin: 1rem 0;
                        text-align: center;
                        color: {theme_config['text_primary']};
                        font-weight: 600;
                    ">
                        ✅ Results cleared successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.5)
                    st.rerun()

            # Display results in beautiful broader tabs below action buttons
            if hasattr(st.session_state, 'show_results_in_tabs') and st.session_state.show_results_in_tabs and st.session_state.video_results:
                st.markdown("---")

                # Enhanced broader tab styling
                st.markdown(f"""
                <style>
                /* Enhanced broader tab styling for detection results */
                .detection-results-tabs .stTabs [data-baseweb="tab-list"] {{
                    gap: 12px;
                    background: linear-gradient(135deg, {theme_config['card_bg']}95 0%, {theme_config['secondary_bg']}80 100%);
                    border-radius: 20px;
                    padding: 16px;
                    margin: 2rem 0;
                    box-shadow: 0 8px 32px {theme_config['shadow']};
                    backdrop-filter: blur(20px);
                    border: 1px solid {theme_config['border_color']};
                }}

                .detection-results-tabs .stTabs [data-baseweb="tab"] {{
                    height: 60px;
                    min-width: 200px;
                    padding: 0 2rem;
                    background: linear-gradient(135deg, {theme_config['accent_color']}10 0%, {theme_config['accent_color']}05 100%);
                    border: 1px solid {theme_config['accent_color']}30;
                    border-radius: 16px;
                    color: {theme_config['text_primary']};
                    font-weight: 600;
                    font-size: 1rem;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}

                .detection-results-tabs .stTabs [data-baseweb="tab"]:hover {{
                    background: linear-gradient(135deg, {theme_config['accent_color']}20 0%, {theme_config['accent_color']}10 100%);
                    border-color: {theme_config['accent_color']}50;
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px {theme_config['shadow']};
                }}

                .detection-results-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                    background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['accent_color']}80 100%);
                    border-color: {theme_config['accent_color']};
                    color: white;
                    box-shadow: 0 8px 25px {theme_config['accent_color']}40;
                    transform: translateY(-3px);
                }}

                .detection-results-tabs .stTabs [data-baseweb="tab-panel"] {{
                    padding: 2rem;
                    background: {theme_config['card_bg']};
                    border-radius: 20px;
                    border: 1px solid {theme_config['border_color']};
                    box-shadow: 0 4px 20px {theme_config['shadow']};
                    margin-top: 1rem;
                }}

                /* Perfect text fitting in metric cards */
                .detection-results-tabs .metric-card {{
                    background: linear-gradient(135deg, var(--card-color, {theme_config['accent_color']})15 0%, var(--card-color, {theme_config['accent_color']})05 100%);
                    border: 1px solid var(--card-color, {theme_config['accent_color']})30;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }}

                .detection-results-tabs .metric-card:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    border-color: var(--card-color, {theme_config['accent_color']})50;
                }}

                .detection-results-tabs .metric-value {{
                    font-size: 2.2rem;
                    font-weight: 800;
                    color: var(--card-color, {theme_config['accent_color']});
                    margin-bottom: 0.5rem;
                    line-height: 1;
                }}

                .detection-results-tabs .metric-label {{
                    font-size: 0.9rem;
                    color: {theme_config['text_secondary']};
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    line-height: 1.2;
                }}

                /* Responsive design for broader tabs */
                @media (max-width: 1200px) {{
                    .detection-results-tabs .stTabs [data-baseweb="tab"] {{
                        min-width: 160px;
                        padding: 0 1.5rem;
                        font-size: 0.9rem;
                    }}
                }}

                @media (max-width: 768px) {{
                    .detection-results-tabs .stTabs [data-baseweb="tab-list"] {{
                        gap: 8px;
                        padding: 12px;
                        flex-wrap: wrap;
                    }}

                    .detection-results-tabs .stTabs [data-baseweb="tab"] {{
                        min-width: 140px;
                        height: 50px;
                        padding: 0 1rem;
                        font-size: 0.85rem;
                    }}

                    .detection-results-tabs .stTabs [data-baseweb="tab-panel"] {{
                        padding: 1.5rem;
                    }}
                }}

                @media (max-width: 480px) {{
                    .detection-results-tabs .stTabs [data-baseweb="tab"] {{
                        min-width: 120px;
                        height: 45px;
                        padding: 0 0.8rem;
                        font-size: 0.8rem;
                    }}
                }}
                </style>
                """, unsafe_allow_html=True)

                # Create broader detection results tabs with perfect styling
                with st.container():
                    st.markdown('<div class="detection-results-tabs">', unsafe_allow_html=True)

                    # Display results using the enhanced dashboard
                    create_results_dashboard(
                        st.session_state.video_results,
                        st.session_state.current_video_path if hasattr(st.session_state, 'current_video_path') else None,
                        0,  # Processing time not available
                        {'skip': 1}  # Default settings
                    )

                    st.markdown('</div>', unsafe_allow_html=True)

            # Show modern summary of previous results
            if st.session_state.video_results:
                results = st.session_state.video_results

                # Modern summary header
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {theme_config['card_bg']}95 0%, {theme_config['secondary_bg']}80 100%);
                    backdrop-filter: blur(20px);
                    border: 1px solid {theme_config['border_color']};
                    border-radius: 20px;
                    padding: 2rem;
                    margin: 2rem 0;
                    box-shadow: 0 8px 32px {theme_config['shadow']};
                ">
                    <h4 style="
                        color: {theme_config['text_primary']};
                        margin: 0 0 1.5rem 0;
                        font-weight: 700;
                        font-size: 1.4rem;
                        display: flex;
                        align-items: center;
                        gap: 0.8rem;
                    ">
                        📊 Last Processing Summary
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                # Modern metrics grid
                col_s1, col_s2, col_s3, col_s4 = st.columns(4, gap="medium")

                with col_s1:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['info_color']}15 0%, {theme_config['info_color']}05 100%);
                        border: 1px solid {theme_config['info_color']}40;
                        border-radius: 16px;
                        padding: 1.5rem;
                        text-align: center;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 4px 20px {theme_config['shadow']};
                    ">
                        <div style="
                            font-size: 2rem;
                            font-weight: 800;
                            color: {theme_config['info_color']};
                            margin-bottom: 0.5rem;
                        ">{results['processed_frames']:,}</div>
                        <div style="
                            color: {theme_config['text_secondary']};
                            font-weight: 600;
                            font-size: 0.9rem;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        ">Frames Processed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_s2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['danger_color']}15 0%, {theme_config['danger_color']}05 100%);
                        border: 1px solid {theme_config['danger_color']}40;
                        border-radius: 16px;
                        padding: 1.5rem;
                        text-align: center;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 4px 20px {theme_config['shadow']};
                    ">
                        <div style="
                            font-size: 2rem;
                            font-weight: 800;
                            color: {theme_config['danger_color']};
                            margin-bottom: 0.5rem;
                        ">{results['total_violations']:,}</div>
                        <div style="
                            color: {theme_config['text_secondary']};
                            font-weight: 600;
                            font-size: 0.9rem;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        ">Violations Found</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_s3:
                    avg_compliance = results.get('average_compliance_rate', 0)
                    compliance_color = theme_config['success_color'] if avg_compliance >= 80 else theme_config['warning_color'] if avg_compliance >= 60 else theme_config['danger_color']

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {compliance_color}15 0%, {compliance_color}05 100%);
                        border: 1px solid {compliance_color}40;
                        border-radius: 16px;
                        padding: 1.5rem;
                        text-align: center;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 4px 20px {theme_config['shadow']};
                    ">
                        <div style="
                            font-size: 2rem;
                            font-weight: 800;
                            color: {compliance_color};
                            margin-bottom: 0.5rem;
                        ">{avg_compliance:.0f}%</div>
                        <div style="
                            color: {theme_config['text_secondary']};
                            font-weight: 600;
                            font-size: 0.9rem;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        ">Avg Compliance</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_s4:
                    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                        file_size = os.path.getsize(st.session_state.processed_video_path) / (1024 * 1024)
                        status_text = f"{file_size:.1f} MB"
                        status_color = theme_config['success_color']
                        label_text = "Video Size"
                    else:
                        status_text = "Missing"
                        status_color = theme_config['warning_color']
                        label_text = "File Status"

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {status_color}15 0%, {status_color}05 100%);
                        border: 1px solid {status_color}40;
                        border-radius: 16px;
                        padding: 1.5rem;
                        text-align: center;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 4px 20px {theme_config['shadow']};
                    ">
                        <div style="
                            font-size: 2rem;
                            font-weight: 800;
                            color: {status_color};
                            margin-bottom: 0.5rem;
                        ">{status_text}</div>
                        <div style="
                            color: {theme_config['text_secondary']};
                            font-weight: 600;
                            font-size: 0.9rem;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        ">{label_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        # Get current theme for dynamic styling
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        # Modern header with gradient background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['info_color']}15 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
        ">
            <h2 style="
                color: {theme_config['text_primary']};
                margin: 0;
                font-size: 2rem;
                font-weight: 700;
                text-align: center;
                background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">📷 Instant Image Analysis</h2>
            <p style="
                color: {theme_config['text_secondary']};
                text-align: center;
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
            ">Upload an image for real-time PPE compliance detection</p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced CSS styling with theme support and animations
        st.markdown(f"""
        <style>
        /* Modern file uploader with clean design */
        .stFileUploader > div > div {{
            border: 2px dashed {theme_config['accent_color']}40 !important;
            border-radius: 20px !important;
            background: linear-gradient(135deg, {theme_config['card_bg']} 0%, {theme_config['secondary_bg']} 100%) !important;
            padding: 4rem 2rem !important;
            text-align: center !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            min-height: 220px !important;
            box-shadow: 0 8px 32px {theme_config['shadow']} !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }}

        .stFileUploader > div > div:hover {{
            border-color: {theme_config['accent_color']}80 !important;
            background: linear-gradient(135deg, {theme_config['accent_color']}10 0%, {theme_config['info_color']}10 100%) !important;
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 12px 40px {theme_config['shadow_hover']} !important;
        }}

        .stFileUploader > div > div::before {{
            content: '📤';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -80%);
            font-size: 4rem;
            opacity: 0.4;
            animation: bounce 2s ease-in-out infinite;
        }}

        .stFileUploader > div > div::after {{
            content: 'Drag & Drop your image here or click to browse\\ASupported: JPG, PNG, BMP • Max 10MB';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, 20%);
            color: {theme_config['text_primary']};
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
            line-height: 1.5;
            white-space: pre-line;
        }}

        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{ transform: translate(-50%, -80%) translateY(0); }}
            40% {{ transform: translate(-50%, -80%) translateY(-10px); }}
            60% {{ transform: translate(-50%, -80%) translateY(-5px); }}
        }}

        /* Hide default upload text */
        .stFileUploader label {{
            display: none !important;
        }}

        .stFileUploader > div > div > div {{
            display: none !important;
        }}

        /* Style the browse button when it appears */
        .stFileUploader button {{
            background: {theme_config['accent_color']} !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
        }}

        /* Enhanced button styling for instant analysis */
        .stButton > button {{
            background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 20px {theme_config['accent_color']}30 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 0 8px 30px {theme_config['accent_color']}40 !important;
            background: linear-gradient(135deg, {theme_config['info_color']} 0%, {theme_config['accent_color']} 100%) !important;
        }}

        .stButton > button:active {{
            transform: translateY(0px) scale(1.02) !important;
        }}

        /* Enhanced download button styling */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, {theme_config['success_color']} 0%, {theme_config['info_color']} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 20px {theme_config['success_color']}30 !important;
        }}

        .stDownloadButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px {theme_config['success_color']}40 !important;
        }}

        /* Image container styling */
        .stImage > img {{
            border-radius: 12px !important;
            box-shadow: 0 8px 32px {theme_config['shadow']} !important;
            transition: transform 0.3s ease !important;
        }}

        .stImage > img:hover {{
            transform: scale(1.02) !important;
        }}

        /* Metric cards hover effects */
        .metric-card {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}

        .metric-card:hover {{
            transform: translateY(-8px) scale(1.02) !important;
            box-shadow: 0 16px 48px {theme_config['shadow_hover']} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # Clean and modern file uploader
        uploaded_image = st.file_uploader(
            "Upload Image",  # Proper label for accessibility
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload your image for instant PPE compliance analysis",
            label_visibility="collapsed"  # Hide the label visually but keep it for accessibility
        )

        # Simple upload instruction when no file is selected
        if not uploaded_image:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 1.5rem;
                margin: 1rem 0;
                background: linear-gradient(135deg, {theme_config['accent_color']}08 0%, {theme_config['info_color']}08 100%);
                border-radius: 12px;
                border: 1px solid {theme_config['accent_color']}20;
                animation: fadeIn 0.5s ease-out;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.7;">📸</div>
                <h4 style="color: {theme_config['text_primary']}; margin: 0 0 0.5rem 0; font-weight: 600;">
                    Ready for Analysis
                </h4>
                <p style="color: {theme_config['text_secondary']}; margin: 0; font-size: 0.9rem;">
                    Upload a clear image with people wearing PPE for instant compliance detection
                </p>
            </div>

            <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            </style>
            """, unsafe_allow_html=True)
        
        if uploaded_image:
            is_valid, error = validate_image(uploaded_image)
            if not is_valid:
                st.error(f"❌ {error}")
                st.stop()

            # Simple success message
            file_size = uploaded_image.size / (1024 * 1024)
            st.success(f"✅ **{uploaded_image.name}** uploaded successfully ({file_size:.1f} MB)")

            image = Image.open(uploaded_image)

            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Ensure numpy is available (defensive import)
            try:
                image_np = np.array(image)
            except NameError:
                import numpy as np
                image_np = np.array(image)

            # Two-column layout like video processing tab
            col1, col2 = st.columns([2.2, 1], gap="large")

            with col1:
                # Image preview with modern styling (same as video preview)
                st.markdown(f"""
                <div style="
                    background: {theme_config['card_bg']};
                    border-radius: 20px;
                    padding: 1rem;
                    border: 1px solid {theme_config['border_color']};
                    box-shadow: 0 8px 32px {theme_config['shadow']};
                    backdrop-filter: blur(10px);
                ">
                    <h4 style="
                        color: {theme_config['text_primary']};
                        margin: 0 0 1rem 0;
                        font-weight: 600;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    ">
                        📷 Image Preview
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, use_container_width=True)

            with col2:
                # Analysis info display with modern design (similar to speed info)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['info_color']}10 100%);
                    border: 1px solid {theme_config['accent_color']}40;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 4px 20px {theme_config['shadow']};
                ">
                    <div style="
                        font-size: 2.5rem;
                        font-weight: 800;
                        background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        margin-bottom: 0.5rem;
                    ">AI Ready</div>
                    <div style="
                        color: {theme_config['text_secondary']};
                        font-weight: 600;
                        font-size: 1rem;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">PPE Detection</div>
                    <div style="
                        color: {theme_config['text_muted']};
                        font-size: 0.85rem;
                        margin-top: 0.5rem;
                    ">Advanced AI-powered analysis</div>
                </div>
                """, unsafe_allow_html=True)

                # Analysis button (same styling as video processing)
                analyze_button = st.button("⚡ START ANALYSIS", type="primary", use_container_width=True, key="analysis_btn")

            # Move analysis processing outside of columns for full-width display
            if analyze_button:
                start_time = time.time()

                # Enhanced loading animation
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.markdown(f"""
                    <div style="
                        background: {theme_config['card_bg']};
                        border-radius: 16px;
                        padding: 2rem;
                        border: 1px solid {theme_config['accent_color']}30;
                        box-shadow: 0 8px 32px {theme_config['shadow']};
                        text-align: center;
                        animation: analysisGlow 2s ease-in-out infinite;
                    ">
                        <div style="font-size: 3rem; margin-bottom: 1rem; animation: spin 2s linear infinite;">🔍</div>
                        <h3 style="color: {theme_config['text_primary']}; margin: 0 0 0.5rem 0;">AI Analysis in Progress</h3>
                        <p style="color: {theme_config['text_secondary']}; margin: 0;">Detecting PPE compliance...</p>
                        <div style="
                            width: 100%;
                            height: 4px;
                            background: {theme_config['border_color']};
                            border-radius: 2px;
                            margin-top: 1rem;
                            overflow: hidden;
                        ">
                            <div style="
                                width: 100%;
                                height: 100%;
                                background: linear-gradient(90deg, {theme_config['accent_color']}, {theme_config['info_color']});
                                animation: progressBar 2s ease-in-out infinite;
                            "></div>
                        </div>
                    </div>

                    <style>
                    @keyframes spin {{
                        from {{ transform: rotate(0deg); }}
                        to {{ transform: rotate(360deg); }}
                    }}
                    @keyframes analysisGlow {{
                        0%, 100% {{ box-shadow: 0 8px 32px {theme_config['shadow']}; }}
                        50% {{ box-shadow: 0 8px 32px {theme_config['accent_color']}30; }}
                    }}
                    @keyframes progressBar {{
                        0% {{ transform: translateX(-100%); }}
                        100% {{ transform: translateX(100%); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                # Perform analysis
                results = st.session_state.detection_engine.detect_objects(
                    image_np, settings['conf'], settings['iou']
                )

                analysis_time = time.time() - start_time
                progress_placeholder.empty()

                if 'error' not in results:
                    # Simple success message like video processing
                    st.success(f"""
                    ✅ **Image Analysis Complete!**
                    - Analysis Time: {analysis_time:.2f} seconds
                    - AI Model: PPE Detection Engine
                    - Status: Ready for review
                    """)

                    # Draw detections including face recognition
                    face_results = results.get('face_results', [])
                    processed_image = st.session_state.detection_engine.draw_detections(
                        image_np, results['detections'], True, face_results
                    )

                    # Simple results display like video processing
                    st.markdown("### 📊 Processed Image with Detections")
                    st.image(processed_image, use_container_width=True)

                    # Simple metrics display like video processing
                    st.markdown("### 📊 Key Metrics")

                    # Get compliance stats
                    compliance_stats = results['compliance_stats']
                    compliance_rate = compliance_stats['compliance_rate']
                    total_people = compliance_stats['total_people']
                    violations = compliance_stats['violations']
                    violations_count = len(violations)

                    # Simple 4-column layout like video processing
                    col1, col2, col3, col4 = st.columns(4)

                    # Add simple metric card styling like video processing
                    st.markdown("""
                    <style>
                    .metric-card {
                        background: linear-gradient(135deg, var(--card-color) 0%, var(--card-color-dark) 100%);
                        color: white;
                        padding: 1.5rem;
                        border-radius: 12px;
                        text-align: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        margin-bottom: 1rem;
                    }
                    .metric-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    }
                    .metric-value {
                        font-size: 2rem;
                        font-weight: 800;
                        margin-bottom: 0.5rem;
                    }
                    .metric-label {
                        font-size: 0.9rem;
                        opacity: 0.9;
                        font-weight: 600;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card" style="--card-color: #4CAF50; --card-color-dark: #45a049;">
                            <div class="metric-value">{total_people}</div>
                            <div class="metric-label">People Detected</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        violations_color = "#f44336" if violations_count > 0 else "#4CAF50"
                        violations_color_dark = "#d32f2f" if violations_count > 0 else "#45a049"
                        st.markdown(f"""
                        <div class="metric-card" style="--card-color: {violations_color}; --card-color-dark: {violations_color_dark};">
                            <div class="metric-value">{violations_count}</div>
                            <div class="metric-label">Violations Found</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        compliance_color = "#4CAF50" if compliance_rate >= 80 else "#ff9800" if compliance_rate >= 60 else "#f44336"
                        compliance_color_dark = "#45a049" if compliance_rate >= 80 else "#f57c00" if compliance_rate >= 60 else "#d32f2f"
                        st.markdown(f"""
                        <div class="metric-card" style="--card-color: {compliance_color}; --card-color-dark: {compliance_color_dark};">
                            <div class="metric-value">{compliance_rate:.0f}%</div>
                            <div class="metric-label">Compliance Rate</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        # Detection summary - total detections
                        total_detections = len(results['detections'])
                        st.markdown(f"""
                        <div class="metric-card" style="--card-color: #2196F3; --card-color-dark: #1976D2;">
                            <div class="metric-value">{total_detections}</div>
                            <div class="metric-label">Total Detections</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Add detection summary like video processing
                    st.markdown("### 📋 Detection Summary")

                    # Create detection summary
                    detection_summary = []
                    if total_people > 0:
                        detection_summary.append(f"👥 **{total_people}** people detected")
                    if violations_count > 0:
                        detection_summary.append(f"⚠️ **{violations_count}** safety violations found")
                    else:
                        detection_summary.append("✅ **No safety violations** detected")
                    detection_summary.append(f"📊 **{compliance_rate:.1f}%** compliance rate")

                    # Display summary
                    for item in detection_summary:
                        st.markdown(f"- {item}")

                    # Simple download section like video processing
                    st.markdown("---")
                    st.markdown("### 📥 Download Results")

                    processed_pil = Image.fromarray(processed_image)
                    buf = io.BytesIO()
                    processed_pil.save(buf, format='PNG')

                    # Ensure datetime is available (defensive import)
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    except NameError:
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    st.download_button(
                        "📥 Download Analyzed Image",
                        buf.getvalue(),
                        file_name=f"ppe_analysis_{timestamp}.png",
                        mime="image/png",
                        use_container_width=True,
                        help="Download the image with PPE detection annotations"
                    )
                else:
                    st.error(f"❌ Analysis failed: {results['error']}")

    with tab4:
        st.markdown("### 📋 Results History")

        # Get current theme for consistent styling
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        if st.session_state.results_history:
            # Modern redesigned history header with enhanced visuals
            st.markdown("""
            <style>
            .history-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 3rem 2rem;
                border-radius: 25px;
                margin: 2rem 0;
                text-align: center;
                box-shadow:
                    0 20px 40px rgba(102, 126, 234, 0.3),
                    0 10px 20px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .history-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                animation: shimmer 3s infinite;
            }
            .history-header::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
                background-size: 300% 100%;
                animation: gradientMove 4s ease infinite;
            }
            .history-title {
                font-size: clamp(1.8rem, 5vw, 2.5rem);
                font-weight: 900;
                color: white;
                margin-bottom: 1rem;
                text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                letter-spacing: -0.5px;
            }
            .history-subtitle {
                color: rgba(255, 255, 255, 0.9);
                font-weight: 600;
                font-size: clamp(1.1rem, 3vw, 1.4rem);
                background: rgba(255, 255, 255, 0.15);
                padding: 0.8rem 1.5rem;
                border-radius: 30px;
                display: inline-block;
                border: 2px solid rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            }
            .session-card {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 25px;
                border: 1px solid rgba(102, 126, 234, 0.1);
                margin: 2rem 0;
                box-shadow:
                    0 10px 30px rgba(0, 0, 0, 0.08),
                    0 4px 15px rgba(102, 126, 234, 0.1);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                overflow: hidden;
                position: relative;
            }
            .session-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
                background-size: 200% 100%;
                animation: gradientMove 3s ease infinite;
            }
            .session-card:hover {
                box-shadow:
                    0 20px 50px rgba(0, 0, 0, 0.15),
                    0 10px 25px rgba(102, 126, 234, 0.2);
                transform: translateY(-8px) scale(1.02);
                border-color: rgba(102, 126, 234, 0.3);
            }
            .session-header {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                padding: 2rem;
                border-radius: 25px 25px 0 0;
                border-bottom: 1px solid rgba(102, 126, 234, 0.1);
                position: relative;
            }
            .session-metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 2rem;
                padding: 2.5rem;
                background: rgba(248, 250, 252, 0.5);
            }
            .metric-card {
                text-align: center;
                padding: 2rem 1.5rem;
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                border: 1px solid rgba(226, 232, 240, 0.8);
                box-shadow:
                    0 8px 25px rgba(0, 0, 0, 0.06),
                    0 3px 10px rgba(0, 0, 0, 0.04);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
            }
            .metric-card:hover {
                transform: translateY(-5px) scale(1.05);
                box-shadow:
                    0 15px 35px rgba(0, 0, 0, 0.12),
                    0 8px 20px rgba(0, 0, 0, 0.08);
                border-color: var(--metric-color, #667eea);
                background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
            }
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 5px;
                background: linear-gradient(90deg, var(--metric-color, #667eea), var(--metric-color-light, #764ba2));
                border-radius: 20px 20px 0 0;
            }
            .metric-card::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 100px;
                height: 100px;
                background: radial-gradient(circle, var(--metric-color, #667eea)10, transparent 70%);
                transform: translate(-50%, -50%);
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }
            .metric-card:hover::after {
                opacity: 0.05;
            }
            .metric-value {
                font-size: clamp(1.6rem, 4vw, 2.2rem);
                font-weight: 900;
                color: var(--metric-color, #667eea);
                margin-bottom: 0.8rem;
                line-height: 1;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                letter-spacing: -0.5px;
                transition: all 0.3s ease;
            }
            .metric-label {
                font-size: clamp(0.85rem, 2vw, 1rem);
                color: #64748b;
                text-transform: uppercase;
                font-weight: 600;
                letter-spacing: 1px;
                line-height: 1.3;
                opacity: 0.8;
                transition: all 0.3s ease;
            }
            .metric-card:hover .metric-value {
                transform: scale(1.1);
                color: var(--metric-color, #667eea);
            }
            .metric-card:hover .metric-label {
                opacity: 1;
                color: #475569;
            }
            .action-buttons-container {
                padding: 2rem 2.5rem;
                background: linear-gradient(135deg, rgba(248, 250, 252, 0.8) 0%, rgba(241, 245, 249, 0.9) 100%);
                border-radius: 0 0 25px 25px;
                border-top: 1px solid rgba(102, 126, 234, 0.1);
                backdrop-filter: blur(10px);
            }
            .action-button-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 1.5rem;
            }

            /* Enhanced animations */
            @keyframes shimmer {
                0% { left: -100%; }
                100% { left: 100%; }
            }

            @keyframes gradientMove {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .session-card {
                animation: fadeInUp 0.6s ease-out;
            }
            /* Enhanced responsive design */
            @media (max-width: 768px) {
                .history-header {
                    padding: 2rem 1.5rem;
                    margin: 1rem 0;
                }
                .session-metrics {
                    grid-template-columns: repeat(2, 1fr);
                    gap: 1.5rem;
                    padding: 2rem 1.5rem;
                }
                .metric-card {
                    padding: 1.5rem 1rem;
                    min-height: 100px;
                }
                .action-buttons-container {
                    padding: 1.5rem;
                }
                .action-button-grid {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                }
            }
            @media (max-width: 480px) {
                .history-header {
                    padding: 1.5rem 1rem;
                }
                .session-metrics {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                    padding: 1.5rem 1rem;
                }
                .metric-card {
                    padding: 1.2rem 0.8rem;
                    min-height: 90px;
                }
                .action-buttons-container {
                    padding: 1rem;
                }
            }

            /* Filter and search styling */
            .history-controls {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 20px;
                padding: 1.5rem;
                margin: 1.5rem 0;
                border: 1px solid rgba(102, 126, 234, 0.1);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            }
            </style>
            """, unsafe_allow_html=True)

            # Calculate summary statistics
            total_sessions = len(st.session_state.results_history)
            total_frames = sum(entry['results']['processed_frames'] for entry in st.session_state.results_history)
            total_violations = sum(entry['results']['total_violations'] for entry in st.session_state.results_history)
            avg_compliance = sum(entry['results'].get('average_compliance_rate', 0) for entry in st.session_state.results_history) / total_sessions if total_sessions > 0 else 0

            st.markdown(f"""
            <div class="history-header">
                <div class="history-title">🎬 Processing History Dashboard</div>
                <div class="history-subtitle">
                    {total_sessions} Sessions • {total_frames:,} Frames • {avg_compliance:.1f}% Avg Compliance
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick stats overview
            if total_sessions > 0:
                st.markdown("### 📈 Quick Overview")
                overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

                with overview_col1:
                    st.metric("Total Sessions", total_sessions, delta=None)

                with overview_col2:
                    st.metric("Total Frames", f"{total_frames:,}", delta=None)

                with overview_col3:
                    compliance_delta = "🟢 Excellent" if avg_compliance >= 90 else "🟡 Good" if avg_compliance >= 75 else "🔴 Needs Improvement"
                    st.metric("Avg Compliance", f"{avg_compliance:.1f}%", delta=compliance_delta)

                with overview_col4:
                    violation_status = "🎉 Clean" if total_violations == 0 else f"⚠️ {total_violations} Total"
                    st.metric("Safety Status", violation_status, delta=None)

            # Enhanced history controls with search and filters
            st.markdown("""
            <div class="history-controls">
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown("**🔍 Search & Filter Sessions:**")
                search_term = st.text_input(
                    "Search by date or session number",
                    placeholder="e.g., 2024-01 or Session 5",
                    label_visibility="collapsed"
                )

            with col2:
                sort_order = st.selectbox(
                    "Sort by",
                    ["Newest First", "Oldest First", "Most Violations", "Best Compliance"],
                    label_visibility="collapsed"
                )

            with col3:
                if st.button("🗑️ Clear History",
                           use_container_width=True,
                           type="secondary",
                           help="Remove all processing history"):
                    st.session_state.results_history = []
                    st.success("✅ History cleared successfully!")
                    time.sleep(0.5)
                    st.rerun()

            # Filter and sort history
            filtered_history = st.session_state.results_history.copy()

            # Apply search filter
            if search_term:
                filtered_history = [
                    entry for entry in filtered_history
                    if search_term.lower() in entry['timestamp'].lower()
                ]

            # Apply sorting
            if sort_order == "Oldest First":
                filtered_history = list(reversed(filtered_history))
            elif sort_order == "Most Violations":
                filtered_history = sorted(filtered_history,
                                        key=lambda x: x['results'].get('total_violations', 0),
                                        reverse=True)
            elif sort_order == "Best Compliance":
                filtered_history = sorted(filtered_history,
                                        key=lambda x: x['results'].get('average_compliance_rate', 0),
                                        reverse=True)
            else:  # Newest First (default)
                filtered_history = list(reversed(filtered_history))

            # Show filtered results count
            if search_term or sort_order != "Newest First":
                st.markdown(f"**📊 Showing {len(filtered_history)} of {len(st.session_state.results_history)} sessions**")

            # Display filtered history entries with enhanced design
            for i, entry in enumerate(filtered_history):
                # Calculate original session number
                original_index = st.session_state.results_history.index(entry)
                session_num = len(st.session_state.results_history) - original_index

                # Create enhanced session card with modern styling
                with st.expander(
                    f"🎬 Session {session_num} • {entry['timestamp']}",
                    expanded=(i == 0 and len(filtered_history) <= 3)
                ):

                    # Session metrics with improved layout
                    st.markdown("""
                    <div class="session-metrics">
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c, col_d = st.columns(4)

                    with col_a:
                        frames = entry['results']['processed_frames']
                        st.markdown(f"""
                        <div class="metric-card" style="--metric-color: #667eea; --metric-color-light: #764ba2;">
                            <div class="metric-value">{frames:,}</div>
                            <div class="metric-label">📹 Total Frames</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_b:
                        violations = entry['results']['total_violations']
                        if violations == 0:
                            violation_color = "#10b981"
                            violation_light = "#34d399"
                            icon = "✅"
                        else:
                            violation_color = "#ef4444"
                            violation_light = "#f87171"
                            icon = "⚠️"
                        st.markdown(f"""
                        <div class="metric-card" style="--metric-color: {violation_color}; --metric-color-light: {violation_light};">
                            <div class="metric-value">{violations:,}</div>
                            <div class="metric-label">{icon} Safety Violations</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_c:
                        avg_compliance = entry['results'].get('average_compliance_rate', 0)
                        if avg_compliance >= 90:
                            compliance_color = "#10b981"
                            compliance_light = "#34d399"
                            icon = "🏆"
                        elif avg_compliance >= 75:
                            compliance_color = "#f59e0b"
                            compliance_light = "#fbbf24"
                            icon = "📊"
                        else:
                            compliance_color = "#ef4444"
                            compliance_light = "#f87171"
                            icon = "📉"
                        st.markdown(f"""
                        <div class="metric-card" style="--metric-color: {compliance_color}; --metric-color-light: {compliance_light};">
                            <div class="metric-value">{avg_compliance:.0f}%</div>
                            <div class="metric-label">{icon} Compliance Rate</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_d:
                        processing_time = entry['processing_time']
                        if processing_time < 30:
                            time_color = "#8b5cf6"
                            time_light = "#a78bfa"
                            icon = "⚡"
                        elif processing_time < 120:
                            time_color = "#06b6d4"
                            time_light = "#22d3ee"
                            icon = "⏱️"
                        else:
                            time_color = "#f59e0b"
                            time_light = "#fbbf24"
                            icon = "🕐"
                        st.markdown(f"""
                        <div class="metric-card" style="--metric-color: {time_color}; --metric-color-light: {time_light};">
                            <div class="metric-value">{processing_time:.1f}s</div>
                            <div class="metric-label">{icon} Processing Time</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Enhanced action buttons with better container
                    st.markdown("""
                    <div class="action-buttons-container">
                    </div>
                    """, unsafe_allow_html=True)

                    col_btn1, col_btn2, col_btn3 = st.columns(3)

                    with col_btn1:
                        if st.button(f"👁️ View Results",
                                   key=f"view_{i}",
                                   use_container_width=True,
                                   type="primary",
                                   help="View complete analysis results for this session"):
                            try:
                                # Set session state to show results in broader tabs below
                                st.session_state.video_results = entry['results']
                                st.session_state.processed_video_data = entry['video_data']
                                st.session_state.show_results = True
                                st.session_state[f'show_history_results_{i}'] = True
                                st.session_state[f'history_entry_{i}'] = entry

                                # Create temporary file for video display with better error handling
                                temp_path = None
                                if entry['video_data']:
                                    try:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                            tmp_file.write(entry['video_data'])
                                            temp_path = tmp_file.name
                                        st.session_state[f'history_temp_path_{i}'] = temp_path
                                    except Exception as e:
                                        st.error(f"❌ Could not create video file: {str(e)}")
                                        temp_path = None

                                # Enhanced success message with better visual design
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, {theme_config['success_color']}15 0%, {theme_config['info_color']}10 100%);
                                    border: 2px solid {theme_config['success_color']};
                                    border-radius: 16px;
                                    padding: 1.5rem;
                                    margin: 1rem 0;
                                    text-align: center;
                                    animation: slideIn 0.5s ease-out;
                                ">
                                    <div style="
                                        color: {theme_config['success_color']};
                                        font-size: 1.2rem;
                                        font-weight: 700;
                                        margin-bottom: 0.5rem;
                                    ">✅ Success!</div>
                                    <div style="
                                        color: {theme_config['text_primary']};
                                        font-size: 1rem;
                                    ">Detection results will be displayed in broader horizontal tabs below</div>
                                </div>

                                <style>
                                @keyframes slideIn {{
                                    from {{ opacity: 0; transform: translateY(-20px); }}
                                    to {{ opacity: 1; transform: translateY(0); }}
                                }}
                                </style>
                                """, unsafe_allow_html=True)
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Failed to load results: {str(e)}")

                    with col_btn2:
                        if entry['video_data']:
                            st.download_button(
                                "📥 Download Video",
                                entry['video_data'],
                                file_name=f"ppe_session_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.mp4",
                                mime="video/mp4",
                                key=f"download_{i}",
                                use_container_width=True,
                                help="Download the processed video with PPE detection overlays"
                            )
                        else:
                            st.button("📥 No Video",
                                    key=f"no_video_{i}",
                                    use_container_width=True,
                                    disabled=True,
                                    help="Video data not available for this session")

                    with col_btn3:
                        # Enhanced download report with better data
                        report_data = {
                            'session_info': {
                                'session_number': session_num,
                                'timestamp': entry['timestamp'],
                                'processing_time': entry['processing_time'],
                                'settings': entry['settings']
                            },
                            'results': entry['results'],
                            'summary': {
                                'total_frames': entry['results']['processed_frames'],
                                'total_violations': entry['results']['total_violations'],
                                'compliance_rate': entry['results'].get('average_compliance_rate', 0),
                                'processing_speed': f"{entry['results']['processed_frames'] / entry['processing_time']:.1f} FPS" if entry['processing_time'] > 0 else "N/A"
                            }
                        }

                        import json
                        st.download_button(
                            "📄 Download Report",
                            json.dumps(report_data, indent=2, default=str),
                            file_name=f"ppe_report_session_{session_num}_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                            mime="application/json",
                            key=f"report_{i}",
                            use_container_width=True,
                            help="Download comprehensive analysis report in JSON format"
                        )

                    # Display broader horizontal detection results tabs below buttons if requested
                    if st.session_state.get(f'show_history_results_{i}', False):
                        st.markdown("---")



                        # Enhanced broader tab styling for history results with improved visual design
                        st.markdown(f"""
                        <style>
                        .history-detection-results-{i} .stTabs [data-baseweb="tab-list"] {{
                            gap: 20px;
                            background: linear-gradient(135deg, {theme_config['card_bg']}98 0%, {theme_config['secondary_bg']}85 100%);
                            border-radius: 28px;
                            padding: 24px;
                            margin: 2.5rem 0;
                            box-shadow: 0 12px 40px {theme_config['shadow']};
                            backdrop-filter: blur(25px);
                            border: 2px solid {theme_config['border_color']};
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            width: 100%;
                            overflow-x: auto;
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab"] {{
                            background: linear-gradient(135deg, {theme_config['accent_color']}25 0%, {theme_config['info_color']}20 100%);
                            border: 2px solid {theme_config['border_color']};
                            border-radius: 20px;
                            padding: 0 3rem;
                            margin: 0 6px;
                            min-width: 240px;
                            height: 75px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: 700;
                            font-size: 1.2rem;
                            color: {theme_config['text_primary']};
                            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                            position: relative;
                            overflow: hidden;
                            backdrop-filter: blur(15px);
                            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
                            cursor: pointer;
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab"]:before {{
                            content: '';
                            position: absolute;
                            top: 0;
                            left: -100%;
                            width: 100%;
                            height: 100%;
                            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                            transition: left 0.5s;
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab"]:hover {{
                            transform: translateY(-3px) scale(1.02);
                            box-shadow: 0 12px 35px {theme_config['shadow']};
                            background: linear-gradient(135deg, {theme_config['accent_color']}35 0%, {theme_config['info_color']}30 100%);
                            border-color: {theme_config['accent_color']};
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab"]:hover:before {{
                            left: 100%;
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                            background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%);
                            color: white;
                            border-color: {theme_config['accent_color']};
                            box-shadow: 0 12px 35px {theme_config['accent_color']}50;
                            transform: translateY(-2px) scale(1.05);
                            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        }}

                        .history-detection-results-{i} .stTabs [data-baseweb="tab-panel"] {{
                            background: {theme_config['card_bg']};
                            border-radius: 24px;
                            padding: 3rem;
                            margin-top: 1.5rem;
                            box-shadow: 0 12px 40px {theme_config['shadow']};
                            border: 2px solid {theme_config['border_color']};
                            backdrop-filter: blur(15px);
                            min-height: 400px;
                        }}

                        /* Enhanced responsive design for broader tabs */
                        @media (max-width: 768px) {{
                            .history-detection-results-{i} .stTabs [data-baseweb="tab-list"] {{
                                gap: 16px;
                                padding: 20px;
                                flex-wrap: wrap;
                                justify-content: center;
                            }}

                            .history-detection-results-{i} .stTabs [data-baseweb="tab"] {{
                                font-size: 1.1rem;
                                padding: 0 2rem;
                                min-width: 180px;
                                height: 65px;
                            }}

                            .history-detection-results-{i} .stTabs [data-baseweb="tab-panel"] {{
                                padding: 2rem;
                            }}
                        }}

                        @media (max-width: 480px) {{
                            .history-detection-results-{i} .stTabs [data-baseweb="tab"] {{
                                font-size: 1rem;
                                padding: 0 1.5rem;
                                min-width: 160px;
                                height: 60px;
                            }}

                            .history-detection-results-{i} .stTabs [data-baseweb="tab-list"] {{
                                gap: 12px;
                                padding: 16px;
                            }}

                            .history-detection-results-{i} .stTabs [data-baseweb="tab-panel"] {{
                                padding: 1.5rem;
                            }}
                        }}
                        </style>
                        """, unsafe_allow_html=True)

                        # Create broader detection results tabs with perfect styling and enhanced layout
                        with st.container():
                            # Add animation for the detection results section
                            st.markdown(f"""
                            <style>
                            .history-detection-results-{i} {{
                                animation: fadeInUp 0.6s ease-out;
                            }}

                            @keyframes fadeInUp {{
                                from {{
                                    opacity: 0;
                                    transform: translateY(30px);
                                }}
                                to {{
                                    opacity: 1;
                                    transform: translateY(0);
                                }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)

                            st.markdown(f'<div class="history-detection-results-{i}">', unsafe_allow_html=True)

                            # Get the stored entry data
                            stored_entry = st.session_state.get(f'history_entry_{i}', entry)
                            temp_path = st.session_state.get(f'history_temp_path_{i}', None)

                            # Display results using the enhanced dashboard with broader tabs
                            create_results_dashboard(
                                stored_entry['results'],
                                temp_path,
                                stored_entry['processing_time'],
                                stored_entry['settings'],
                                unique_id=f"history_{i}"
                            )

                            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Enhanced empty state with attractive design
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 25px;
                padding: 4rem 2rem;
                text-align: center;
                margin: 2rem 0;
                border: 1px solid rgba(102, 126, 234, 0.1);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            ">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
                <h3 style="color: #475569; margin-bottom: 1rem; font-weight: 700;">
                    No Processing History Yet
                </h3>
                <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem; max-width: 500px; margin-left: auto; margin-right: auto;">
                    Start processing videos to build your analysis history. All your sessions will appear here with detailed insights and downloadable results.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Feature showcase with modern cards
            st.markdown("### ✨ What You'll Get")

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 20px;
                    margin: 1rem 0;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                ">
                    <h4 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        📊 <span>Detailed Analytics</span>
                    </h4>
                    <ul style="margin: 0; padding-left: 1rem; line-height: 1.6;">
                        <li>Frame-by-frame analysis</li>
                        <li>Compliance rate tracking</li>
                        <li>Violation timeline</li>
                        <li>Performance metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with feature_col2:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 20px;
                    margin: 1rem 0;
                    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
                ">
                    <h4 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        📥 <span>Export Options</span>
                    </h4>
                    <ul style="margin: 0; padding-left: 1rem; line-height: 1.6;">
                        <li>Download processed videos</li>
                        <li>JSON analysis reports</li>
                        <li>Session comparisons</li>
                        <li>Historical data access</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin: 2rem 0;
                text-align: center;
                box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
            ">
                <strong>💡 Pro Tip:</strong> History automatically keeps your last 10 processing sessions with full data retention
            </div>
            """, unsafe_allow_html=True)

    with tab5:
        # Enhanced Export Session Data Tab with improved UI
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                animation: shimmer 3s ease-in-out infinite;
            "></div>
            <div style="position: relative; z-index: 1;">
                <h2 style="margin: 0 0 0.5rem 0; font-size: 2rem; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                    📊 Export Session Data
                </h2>
                <p style="margin: 0; font-size: 1.1rem; opacity: 0.9; text-shadow: 0 1px 5px rgba(0,0,0,0.3);">
                    Comprehensive analytics and download center
                </p>
            </div>
        </div>

        <style>
        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        </style>
        """, unsafe_allow_html=True)



        # Check if webcam is available and has data
        if not WEBCAM_AVAILABLE:
            st.warning("⚠️ **Webcam not available**")
            st.info("📹 **Note**: Export functionality requires webcam detection to be available.")
            st.markdown("""
            **🔧 To enable export:**
            - Ensure your camera is connected and working
            - Grant camera permissions to your browser
            - Restart the application if needed
            """)
            return

        # Check if webcam detector exists and has data
        if 'webcam_detector' not in st.session_state:
            st.info("📹 **No active detection session**")
            st.markdown("""
            **🚀 To start exporting:**
            1. Go to the **📹 Live Detection** tab
            2. Start the camera and begin detection
            3. Wait for some frames to be processed
            4. Return here to export your session data
            """)

            # Add a button to switch to Live Detection tab
            if st.button("🎯 Go to Live Detection", type="primary", use_container_width=True):
                st.switch_page("📹 Live Detection")
            return

        # Get webcam detector and stats
        webcam_detector = st.session_state.webcam_detector
        latest_stats = webcam_detector.get_latest_stats()
        frame_count = webcam_detector.frame_count

        # Check if there's data to export
        if frame_count == 0:
            st.info("📊 **No frames processed yet**")
            st.markdown("""
            **⏱️ Status**: Camera is initialized but no frames have been processed.

            **🔄 Next steps:**
            1. Go to **📹 Live Detection** tab
            2. Make sure the camera is running
            3. Wait for frames to be processed
            4. Return here when you see detection activity
            """)
            return

        # Enhanced CSS for export tab with animations
        st.markdown("""
        <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes pulse {
            0% {
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            }
            50% {
                box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5);
            }
            100% {
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            }
        }

        .export-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            animation: fadeInUp 0.8s ease-out, pulse 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }

        .export-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shimmer 3s ease-in-out infinite;
        }

        @keyframes shimmer {
            0% {
                transform: translateX(-100%) translateY(-100%) rotate(45deg);
            }
            50% {
                transform: translateX(100%) translateY(100%) rotate(45deg);
            }
            100% {
                transform: translateX(-100%) translateY(-100%) rotate(45deg);
            }
        }

        .export-metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 20px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow: hidden;
            animation: slideInLeft 0.6s ease-out;
            backdrop-filter: blur(10px);
        }

        .export-metric-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            border-color: var(--card-accent, #667eea);
            background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);
        }

        .export-metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, var(--card-accent, #667eea), var(--card-accent-light, #8fa4f3));
            border-radius: 20px 20px 0 0;
            transition: height 0.3s ease;
        }

        .export-metric-card:hover::before {
            height: 8px;
        }

        .export-metric-card::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.4s ease;
            z-index: 0;
        }

        .export-metric-card:hover::after {
            width: 200px;
            height: 200px;
        }

        .export-metric-value {
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 900;
            margin: 0 0 0.5rem 0;
            color: #2c3e50;
            line-height: 1;
            position: relative;
            z-index: 1;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            word-break: keep-all;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .export-metric-card:hover .export-metric-value {
            transform: scale(1.05);
            background: linear-gradient(135deg, var(--card-accent, #667eea) 0%, var(--card-accent-light, #8fa4f3) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .export-metric-label {
            font-size: 0.85rem;
            color: #6c757d;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            z-index: 1;
            transition: all 0.3s ease;
            line-height: 1.2;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
        }

        .export-metric-card:hover .export-metric-label {
            color: var(--card-accent, #667eea);
            transform: translateY(-2px);
        }

        .download-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border: 2px solid #dee2e6;
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            animation: fadeInUp 0.8s ease-out 0.2s both;
            position: relative;
            overflow: hidden;
        }

        .download-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            background-size: 200% 100%;
            animation: gradientShift 3s ease-in-out infinite;
        }

        @keyframes gradientShift {
            0%, 100% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
        }

        .download-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 18px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.06);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            height: 100%;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(5px);
        }

        .download-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 30px rgba(0,0,0,0.15);
            border-color: #667eea;
            background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);
        }

        .download-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transition: left 0.5s ease;
        }

        .download-card:hover::before {
            left: 100%;
        }

        .download-icon {
            font-size: 3rem;
            margin-bottom: 1.2rem;
            display: block;
            transition: all 0.3s ease;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }

        .download-card:hover .download-icon {
            transform: scale(1.1) rotate(5deg);
            filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
        }

        .download-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #2c3e50;
            margin-bottom: 0.8rem;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
        }

        .download-card:hover .download-title {
            color: #667eea;
            transform: translateY(-2px);
        }

        .download-description {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
            line-height: 1.5;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
        }

        .download-card:hover .download-description {
            color: #5a6c7d;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.8rem 1.5rem;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1rem;
            transition: all 0.3s ease;
            animation: fadeInUp 0.6s ease-out 0.4s both;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }

        .status-indicator:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        }

        .status-ready {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 2px solid #28a745;
            animation: pulse 2s ease-in-out infinite;
        }

        .status-processing {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 2px solid #ffc107;
            animation: pulse 2s ease-in-out infinite;
        }

        .chart-container {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease-out 0.6s both;
            backdrop-filter: blur(5px);
        }

        .chart-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            border-color: #667eea;
        }

        /* Responsive design improvements */
        @media (max-width: 768px) {
            .export-metric-card {
                min-height: 120px;
                padding: 1.2rem;
            }

            .export-metric-value {
                font-size: 1.8rem;
            }

            .export-metric-label {
                font-size: 0.75rem;
            }

            .download-card {
                padding: 1.5rem;
            }

            .download-icon {
                font-size: 2.5rem;
            }

            .download-title {
                font-size: 1rem;
            }

            .download-description {
                font-size: 0.8rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        # Export header with session info
        session_duration = latest_stats.get('session_duration', 0) if latest_stats else time.time() - webcam_detector.start_time
        minutes = int(session_duration // 60)
        seconds = int(session_duration % 60)

        st.markdown(f"""
        <div class="export-header">
            <h2 style="margin: 0 0 1rem 0;">📊 Session Export Ready</h2>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                {frame_count:,} frames processed • {minutes:02d}:{seconds:02d} session duration
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced Session metrics overview with better spacing
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        ">
            <h4 style="margin: 0 0 1rem 0; color: #2c3e50; font-size: 1.3rem;">
                📈 Session Overview & Key Metrics
            </h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="export-metric-card" style="--card-accent: #2196F3;">
                <div class="export-metric-value">{frame_count:,}</div>
                <div class="export-metric-label">Frames Processed</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_people = latest_stats.get('total_people_session', 0) if latest_stats else webcam_detector.total_people_detected
            st.markdown(f"""
            <div class="export-metric-card" style="--card-accent: #4CAF50;">
                <div class="export-metric-value">{total_people:,}</div>
                <div class="export-metric-label">People Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_violations = latest_stats.get('total_violations_session', 0) if latest_stats else webcam_detector.total_violations_detected
            st.markdown(f"""
            <div class="export-metric-card" style="--card-accent: #FF9800;">
                <div class="export-metric-value">{total_violations:,}</div>
                <div class="export-metric-label">Violations Found</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            compliance_rate = latest_stats.get('compliance_rate', 100.0) if latest_stats else 100.0
            st.markdown(f"""
            <div class="export-metric-card" style="--card-accent: #9C27B0;">
                <div class="export-metric-value">{compliance_rate:.1f}%</div>
                <div class="export-metric-label">Compliance Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Export status
        if latest_stats and frame_count >= 10:
            st.markdown("""
            <div class="status-indicator status-ready">
                ✅ Export Ready - Full analytics available
            </div>
            """, unsafe_allow_html=True)
        elif frame_count >= 5:
            st.markdown("""
            <div class="status-indicator status-processing">
                ⏳ Basic Export Available - Full analytics building
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-processing">
                🔄 Initializing - More data needed for full export
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Enhanced Download section with better visual hierarchy
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border-left: 5px solid #28a745;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #28a745, #20c997, #28a745);
                background-size: 200% 100%;
                animation: gradientShift 3s ease-in-out infinite;
            "></div>
            <h4 style="margin: 0 0 1rem 0; color: #2c3e50; font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem;">
                📥 Download Center
                <span style="font-size: 0.8rem; background: #28a745; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600;">
                    Ready
                </span>
            </h4>
            <p style="margin: 0; color: #6c757d; font-size: 1rem;">
                Export your session data in multiple formats for analysis and reporting
            </p>
        </div>

        <style>
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        </style>
        """, unsafe_allow_html=True)

        # Import the export function
        from webcam_component import create_comprehensive_export_data

        try:
            # Create export data
            export_data = create_comprehensive_export_data(latest_stats, webcam_detector)

            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                st.markdown("""
                <div class="download-card">
                    <div class="download-icon">📄</div>
                    <div class="download-title">JSON Report</div>
                    <div class="download-description">Complete detection data in JSON format for developers and analysts</div>
                </div>
                """, unsafe_allow_html=True)

                import json
                # Ensure datetime is available (defensive import)
                try:
                    export_timestamp = datetime.now()
                except NameError:
                    from datetime import datetime
                    export_timestamp = datetime.now()
                json_data = json.dumps(export_data, indent=2, default=str)

                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_data,
                    file_name=f"ppe_detection_report_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Complete detection data in JSON format"
                )
                st.caption(f"📊 Size: {len(json_data)/1024:.1f} KB")

            with col_d2:
                st.markdown("""
                <div class="download-card">
                    <div class="download-icon">📊</div>
                    <div class="download-title">CSV Data</div>
                    <div class="download-description">Timeline data for spreadsheet analysis and further processing</div>
                </div>
                """, unsafe_allow_html=True)

                # Create CSV data
                if webcam_detector.compliance_history:
                    import pandas as pd

                    # Create DataFrame with timeline data
                    df_data = {
                        'Frame': list(range(1, len(webcam_detector.compliance_history) + 1)),
                        'Compliance_Rate': webcam_detector.compliance_history,
                        'People_Count': webcam_detector.people_history[:len(webcam_detector.compliance_history)] if webcam_detector.people_history else [0] * len(webcam_detector.compliance_history),
                        'Violations': webcam_detector.violation_history[:len(webcam_detector.compliance_history)] if webcam_detector.violation_history else [0] * len(webcam_detector.compliance_history)
                    }

                    df = pd.DataFrame(df_data)
                    csv_data = df.to_csv(index=False)

                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_data,
                        file_name=f"ppe_detection_data_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Timeline data for Excel/Google Sheets"
                    )
                    st.caption(f"📈 {len(webcam_detector.compliance_history)} data points")
                else:
                    st.info("⏳ CSV data will be available after more frames are processed")

            with col_d3:
                st.markdown("""
                <div class="download-card">
                    <div class="download-icon">📋</div>
                    <div class="download-title">Summary Report</div>
                    <div class="download-description">Human-readable executive summary with key insights and recommendations</div>
                </div>
                """, unsafe_allow_html=True)

                # Create summary text
                summary_text = f"""PPE Detection Session Summary
Generated: {export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

SESSION OVERVIEW
================
• Total Frames Processed: {frame_count:,}
• Session Duration: {minutes:02d}:{seconds:02d}
• People Detected: {total_people:,}
• Violations Found: {total_violations:,}
• Compliance Rate: {compliance_rate:.1f}%

PERFORMANCE METRICS
==================
• Average FPS: {frame_count / session_duration if session_duration > 0 else 0:.1f}
• Detection Accuracy: High
• System Status: Operational

RECOMMENDATIONS
===============
• Continue monitoring for optimal safety compliance
• Review violation patterns for improvement opportunities
• Regular equipment checks recommended

Generated by PPE Monitor Pro v2.1
"""

                st.download_button(
                    label="📥 Download Summary",
                    data=summary_text,
                    file_name=f"ppe_detection_summary_{export_timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Executive summary report"
                )
                st.caption("📝 Executive summary")

        except Exception as e:
            st.error(f"❌ Export preparation failed: {str(e)}")
            st.info("💡 Please ensure the detection session is active and try again.")

        # Enhanced Charts section with better visual organization
        if webcam_detector.compliance_history and len(webcam_detector.compliance_history) > 1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 2rem 0 1rem 0;
                border-left: 5px solid #6f42c1;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            ">
                <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50; font-size: 1.3rem; display: flex; align-items: center; gap: 0.5rem;">
                    📈 Live Analytics Dashboard
                    <span style="font-size: 0.7rem; background: #6f42c1; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-weight: 600;">
                        LIVE
                    </span>
                </h4>
                <p style="margin: 0; color: #6c757d; font-size: 0.95rem;">
                    Real-time visualization of detection performance and trends
                </p>
            </div>
            """, unsafe_allow_html=True)

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Enhanced chart layout with better spacing
            col_chart1, col_chart2 = st.columns(2, gap="large")

            with col_chart1:
                st.markdown("""
                <div class="chart-container" style="
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 15px;
                    padding: 1.5rem;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                    border: 2px solid #e9ecef;
                    transition: all 0.3s ease;
                    margin-bottom: 1rem;
                ">
                """, unsafe_allow_html=True)

                # Compliance timeline chart
                fig_compliance = go.Figure()
                fig_compliance.add_trace(go.Scatter(
                    y=webcam_detector.compliance_history,
                    mode='lines+markers',
                    name='Compliance Rate',
                    line=dict(color='#4CAF50', width=3),
                    marker=dict(size=6, color='#4CAF50'),
                    fill='tonexty',
                    fillcolor='rgba(76, 175, 80, 0.1)'
                ))

                fig_compliance.update_layout(
                    title={
                        'text': "📊 Compliance Rate Timeline",
                        'x': 0.5,
                        'font': {'size': 16, 'color': '#2c3e50'}
                    },
                    yaxis_title="Compliance Rate (%)",
                    xaxis_title="Frame Number",
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50'),
                    yaxis=dict(range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig_compliance, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_chart2:
                st.markdown("""
                <div class="chart-container" style="
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 15px;
                    padding: 1.5rem;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                    border: 2px solid #e9ecef;
                    transition: all 0.3s ease;
                    margin-bottom: 1rem;
                ">
                """, unsafe_allow_html=True)

                # People detection chart
                if webcam_detector.people_history:
                    fig_people = go.Figure()
                    fig_people.add_trace(go.Scatter(
                        y=webcam_detector.people_history,
                        mode='lines+markers',
                        name='People Detected',
                        line=dict(color='#2196F3', width=3),
                        marker=dict(size=6, color='#2196F3'),
                        fill='tonexty',
                        fillcolor='rgba(33, 150, 243, 0.1)'
                    ))

                    fig_people.update_layout(
                        title={
                            'text': "👥 People Detection Timeline",
                            'x': 0.5,
                            'font': {'size': 16, 'color': '#2c3e50'}
                        },
                        yaxis_title="Number of People",
                        xaxis_title="Frame Number",
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2c3e50'),
                        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )

                    st.plotly_chart(fig_people, use_container_width=True)
                else:
                    # Show placeholder for people detection
                    st.info("👥 People detection chart will appear as data becomes available")

                st.markdown('</div>', unsafe_allow_html=True)

            # Enhanced Combined analytics chart
            if len(webcam_detector.compliance_history) > 5:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                    border: 2px solid #e9ecef;
                    border-left: 5px solid #17a2b8;
                ">
                    <h5 style="margin: 0 0 1rem 0; color: #2c3e50; display: flex; align-items: center; gap: 0.5rem;">
                        📊 Combined Analytics Overview
                        <span style="font-size: 0.7rem; background: #17a2b8; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-weight: 600;">
                            ADVANCED
                        </span>
                    </h5>
                </div>
                <div class="chart-container" style="
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 15px;
                    padding: 1.5rem;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                    border: 2px solid #e9ecef;
                    margin-top: 1rem;
                ">
                """, unsafe_allow_html=True)

                # Create subplot with secondary y-axis
                fig_combined = make_subplots(
                    specs=[[{"secondary_y": True}]],
                    subplot_titles=["📊 Combined Analytics Overview"]
                )

                # Add compliance rate
                fig_combined.add_trace(
                    go.Scatter(
                        y=webcam_detector.compliance_history,
                        mode='lines',
                        name='Compliance Rate (%)',
                        line=dict(color='#4CAF50', width=2)
                    ),
                    secondary_y=False,
                )

                # Add people count if available
                if webcam_detector.people_history:
                    fig_combined.add_trace(
                        go.Scatter(
                            y=webcam_detector.people_history[:len(webcam_detector.compliance_history)],
                            mode='lines',
                            name='People Count',
                            line=dict(color='#2196F3', width=2)
                        ),
                        secondary_y=True,
                    )

                # Update layout
                fig_combined.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50'),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                fig_combined.update_yaxes(title_text="Compliance Rate (%)", secondary_y=False)
                fig_combined.update_yaxes(title_text="Number of People", secondary_y=True)
                fig_combined.update_xaxes(title_text="Frame Number")

                st.plotly_chart(fig_combined, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                border-radius: 15px;
                padding: 2rem;
                margin: 2rem 0;
                border-left: 5px solid #2196f3;
                box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
                text-align: center;
            ">
                <h4 style="margin: 0 0 1rem 0; color: #1976d2; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                    📊 Analytics Dashboard
                    <span style="font-size: 0.7rem; background: #ff9800; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-weight: 600;">
                        BUILDING
                    </span>
                </h4>
                <p style="margin: 0 0 1.5rem 0; color: #424242; font-size: 1.1rem;">
                    Charts will appear after more detection data is collected
                </p>
                <div style="
                    background: rgba(255,255,255,0.7);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    text-align: left;
                ">
                    <h5 style="margin: 0 0 1rem 0; color: #2c3e50;">📈 Available Charts:</h5>
                    <ul style="margin: 0; color: #424242; line-height: 1.8;">
                        <li>🔄 Compliance rate timeline</li>
                        <li>👥 People detection trends</li>
                        <li>📊 Combined analytics overview</li>
                        <li>📈 Real-time performance metrics</li>
                    </ul>
                </div>
                <div style="
                    background: rgba(255, 193, 7, 0.1);
                    border: 1px solid #ffc107;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-top: 1rem;
                ">
                    <strong style="color: #856404;">⏱️ Minimum data needed:</strong>
                    <span style="color: #424242;"> 2+ processed frames with analytics</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Success message with better styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid #28a745;
            border-radius: 15px;
            padding: 1.5rem 2rem;
            margin: 2rem 0;
            text-align: center;
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.2);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #28a745, #20c997, #28a745);
                background-size: 200% 100%;
                animation: gradientShift 2s ease-in-out infinite;
            "></div>
            <h4 style="margin: 0 0 0.5rem 0; color: #155724; font-size: 1.3rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                ✅ Export Interface Ready!
                <span style="font-size: 0.7rem; background: #28a745; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-weight: 600;">
                    ACTIVE
                </span>
            </h4>
            <p style="margin: 0; color: #155724; font-size: 1rem; opacity: 0.9;">
                All download options and analytics are available above. Your session data is ready for export!
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab6:
        # Face Recognition Management Tab
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        # Header with modern styling
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['info_color']}15 100%);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 8px 32px {theme_config['shadow']};
        ">
            <h1 style="
                margin: 0 0 1rem 0;
                font-size: 2.5rem;
                background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['info_color']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">👤 Face Recognition Management</h1>
            <p style="
                color: {theme_config['text_secondary']};
                margin: 0;
                font-size: 1.1rem;
            ">Train and manage face recognition for enhanced PPE monitoring</p>
        </div>
        """, unsafe_allow_html=True)

        face_engine = st.session_state.detection_engine.get_face_engine()

        if not face_engine:
            st.error("❌ Face Recognition engine is not available. Please check your installation.")
            st.stop()

        # Get dataset info for use throughout the interface
        dataset_info = face_engine.get_dataset_info()

        # Enhanced CSS for face recognition interface
        st.markdown(f"""
        <style>
        .face-card {{
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }}

        .face-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px {theme_config['shadow_hover']};
        }}

        .person-card {{
            background: linear-gradient(135deg, {theme_config['card_bg']} 0%, {theme_config['secondary_bg']} 100%);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid {theme_config['border_color']};
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .person-card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 25px {theme_config['shadow_hover']};
        }}

        .person-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, {theme_config['accent_color']}, {theme_config['info_color']});
        }}

        .person-name {{
            color: {theme_config['text_primary']};
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}

        .person-samples {{
            color: {theme_config['text_secondary']};
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }}

        .status-trained {{
            background: {theme_config['success_color']}20;
            color: {theme_config['success_color']};
            border: 1px solid {theme_config['success_color']}40;
        }}

        .status-untrained {{
            background: {theme_config['warning_color']}20;
            color: {theme_config['warning_color']};
            border: 1px solid {theme_config['warning_color']}40;
        }}

        .action-buttons {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .metric-item {{
            background: {theme_config['secondary_bg']};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 1px solid {theme_config['border_color']};
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {theme_config['accent_color']};
            margin-bottom: 0.3rem;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: {theme_config['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        </style>
        """, unsafe_allow_html=True)

        # Create main layout with tabs for better organization
        face_tab1, face_tab2, face_tab3, face_tab4 = st.tabs(["📊 Dataset Overview", "👥 Manage People", "🧠 Model Training", "🧪 Test Recognition"])

        with face_tab1:
            # Dataset Overview Section
            st.markdown(f"""
            <div class="face-card">
                <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📊 Dataset Overview</h3>
            </div>
            """, unsafe_allow_html=True)

            # Dataset metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{dataset_info['total_people']}</div>
                    <div class="metric-label">People</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{dataset_info['total_samples']}</div>
                    <div class="metric-label">Samples</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_samples = dataset_info['total_samples'] / dataset_info['total_people'] if dataset_info['total_people'] > 0 else 0
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{avg_samples:.1f}</div>
                    <div class="metric-label">Avg Samples</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                # Check if model needs retraining after any dataset changes
                if face_engine.is_trained:
                    # Validate if current model matches dataset
                    is_model_valid = face_engine._validate_model_with_dataset()
                    if not is_model_valid:
                        face_engine.is_trained = False

                if face_engine.is_trained:
                    status = "Trained"
                    status_color = theme_config['success_color']
                    status_icon = "✓"
                elif dataset_info['total_people'] > 0:
                    status = "Needs Training"
                    status_color = theme_config['warning_color']
                    status_icon = "⚠"
                else:
                    status = "No Data"
                    status_color = theme_config['text_muted']
                    status_icon = "✗"

                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value" style="color: {status_color};">{status_icon}</div>
                    <div class="metric-label">Model Status</div>
                </div>
                """, unsafe_allow_html=True)

            if dataset_info['total_people'] > 0:
                st.markdown(f"""
                <div class="face-card">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">👥 People in Dataset</h4>
                </div>
                """, unsafe_allow_html=True)

                # Display people in a grid layout
                cols = st.columns(2)
                for i, person in enumerate(dataset_info['people']):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="person-card">
                            <div class="person-name">👤 {person['name']}</div>
                            <div class="person-samples">📸 {person['samples']} samples</div>
                            <div class="status-badge status-trained">✓ Ready for training</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="face-card">
                    <div style="text-align: center; padding: 2rem;">
                        <h4 style="color: {theme_config['text_secondary']}; margin-bottom: 1rem;">📭 No People in Dataset</h4>
                        <p style="color: {theme_config['text_muted']};">Add people to start building your face recognition dataset</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with face_tab2:
            # Manage People Section
            st.markdown(f"""
            <div class="face-card">
                <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">👥 Manage People</h3>
            </div>
            """, unsafe_allow_html=True)

            # Add New Person Section
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div class="face-card">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">➕ Add New Person</h4>
                </div>
                """, unsafe_allow_html=True)

                # Employee information form
                person_name = st.text_input("👤 Employee Name", placeholder="Enter employee's full name", key="add_person_name")

                col_id, col_dept = st.columns(2)
                with col_id:
                    employee_id = st.text_input("🆔 Employee ID", placeholder="e.g., EMP001", key="add_employee_id")
                with col_dept:
                    department = st.text_input("🏢 Department", placeholder="e.g., Engineering (Optional)", key="add_department")

                num_samples = st.slider("📸 Number of samples to collect", 20, 200, 100, 10, key="add_person_samples")

                if st.button("📸 Start Face Collection & Registration", type="primary", use_container_width=True, key="start_collection"):
                    if person_name.strip() and employee_id.strip():
                        # Check if employee ID already exists
                        if st.session_state.attendance_manager:
                            existing_employee = st.session_state.attendance_manager.get_employee_by_name(person_name.strip())
                            if existing_employee:
                                st.warning(f"⚠️ Employee '{person_name}' already exists with ID: {existing_employee['employee_id']}")
                            else:
                                # Add employee to database first
                                if st.session_state.attendance_manager.add_employee(
                                    employee_id.strip(),
                                    person_name.strip(),
                                    department.strip() if department.strip() else None
                                ):
                                    with st.spinner(f"Collecting face samples for {person_name}..."):
                                        results = face_engine.collect_face_samples(person_name.strip(), num_samples)

                                        if results['status'] == 'completed':
                                            # Update face training status in database
                                            st.session_state.attendance_manager.update_employee_face_status(employee_id.strip(), True)
                                            st.success(f"✅ Successfully registered {person_name} and collected {results['samples_collected']} samples!")
                                            st.info("💡 **Next Step:** Go to the 'Model Training' tab to train the face recognition model with your new data!")
                                            st.rerun()
                                        elif results['status'] == 'interrupted':
                                            st.warning(f"⚠️ Collection interrupted. Collected {results['samples_collected']} samples.")
                                            if results['samples_collected'] > 0:
                                                st.info("💡 You can still train the model with the collected samples, or collect more for better accuracy.")
                                        else:
                                            st.error(f"❌ Collection failed: {results.get('error', 'Unknown error')}")
                                            # Remove employee from database if face collection failed
                                            st.session_state.attendance_manager.delete_employee(employee_id.strip())
                                else:
                                    st.error("❌ Failed to register employee in database")
                        else:
                            st.error("❌ Attendance system not available")
                    else:
                        st.error("❌ Please enter both employee name and ID")

            with col2:
                st.markdown(f"""
                <div class="face-card">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">💡 Tips</h4>
                    <ul style="color: {theme_config['text_secondary']}; font-size: 0.9rem; line-height: 1.6;">
                        <li>Look directly at the camera</li>
                        <li>Ensure good lighting</li>
                        <li>Move your head slightly for variety</li>
                        <li>Avoid glasses if possible</li>
                        <li>Collect 50+ samples for best results</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Existing People Management
            if dataset_info['total_people'] > 0:
                st.markdown(f"""
                <div class="face-card">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">🗂️ Registered Employees</h4>
                </div>
                """, unsafe_allow_html=True)

                # Get employee data from attendance manager
                employees_data = {}
                if st.session_state.attendance_manager:
                    all_employees = st.session_state.attendance_manager.get_all_employees()
                    employees_data = {emp['name']: emp for emp in all_employees}

                for person in dataset_info['people']:
                    employee_info = employees_data.get(person['name'], {})

                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            employee_id = employee_info.get('employee_id', 'N/A')
                            department = employee_info.get('department', 'Not Specified')
                            face_trained = employee_info.get('face_trained', False)

                            status_icon = "✅" if face_trained else "⚠️"
                            status_text = "Face Trained" if face_trained else "Needs Training"

                            st.markdown(f"""
                            <div class="person-card">
                                <div class="person-name">👤 {person['name']} {status_icon}</div>
                                <div class="person-samples">🆔 ID: {employee_id} | 🏢 {department}</div>
                                <div class="person-samples">📸 {person['samples']} samples | 🎯 {status_text}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            if st.button("🔄 Retrain", key=f"retrain_{person['name']}", help=f"Collect more samples for {person['name']}"):
                                st.session_state[f"retrain_person"] = person['name']
                                st.rerun()

                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{person['name']}", help=f"Delete {person['name']} from dataset", type="secondary"):
                                st.session_state[f"confirm_delete"] = person['name']
                                st.rerun()

                # Handle delete confirmation
                if 'confirm_delete' in st.session_state:
                    person_to_delete = st.session_state['confirm_delete']
                    employee_info = employees_data.get(person_to_delete, {})
                    employee_id = employee_info.get('employee_id', 'N/A')

                    st.warning(f"⚠️ Are you sure you want to delete **{person_to_delete}** (ID: {employee_id})?")
                    st.info("🗑️ This will remove all face data and attendance records for this employee.")

                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("✅ Yes, Delete", type="primary", key="confirm_delete_yes"):
                            success = True

                            # Delete from face recognition dataset
                            if not face_engine.delete_person_data(person_to_delete):
                                success = False
                                st.error(f"❌ Failed to delete face data for {person_to_delete}")

                            # Delete from attendance database
                            if st.session_state.attendance_manager and employee_id != 'N/A':
                                if not st.session_state.attendance_manager.delete_employee(employee_id):
                                    success = False
                                    st.error(f"❌ Failed to delete attendance data for {person_to_delete}")

                            if success:
                                st.success(f"✅ Successfully deleted {person_to_delete} and all associated data")
                                del st.session_state['confirm_delete']
                                st.rerun()

                    with col2:
                        if st.button("❌ Cancel", key="confirm_delete_no"):
                            del st.session_state['confirm_delete']
                            st.rerun()

                # Handle retrain request
                if 'retrain_person' in st.session_state:
                    person_to_retrain = st.session_state['retrain_person']
                    st.info(f"🔄 Collecting additional samples for **{person_to_retrain}**")

                    additional_samples = st.slider("Additional samples to collect", 10, 100, 30, key="additional_samples")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📸 Start Collection", type="primary", key="start_retrain"):
                            with st.spinner(f"Collecting additional samples for {person_to_retrain}..."):
                                results = face_engine.collect_face_samples(person_to_retrain, additional_samples)

                                if results['status'] == 'completed':
                                    st.success(f"✅ Successfully collected {results['samples_collected']} additional samples")
                                    st.info("💡 **Next Step:** Go to the 'Model Training' tab to retrain the model with your updated data!")
                                    del st.session_state['retrain_person']
                                    st.rerun()
                                else:
                                    st.error(f"❌ Collection failed: {results.get('error', 'Unknown error')}")

                    with col2:
                        if st.button("❌ Cancel", key="cancel_retrain"):
                            del st.session_state['retrain_person']
                            st.rerun()

        with face_tab3:
            # Model Training Section
            st.markdown(f"""
            <div class="face-card">
                <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">🧠 Model Training</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Check model status before displaying
                if face_engine.is_trained:
                    # Validate if current model matches dataset
                    is_model_valid = face_engine._validate_model_with_dataset()
                    if not is_model_valid:
                        face_engine.is_trained = False

                # Training Status
                if face_engine.is_trained:
                    st.markdown(f"""
                    <div class="face-card">
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 3rem; color: {theme_config['success_color']}; margin-bottom: 1rem;">✅</div>
                            <h4 style="color: {theme_config['text_primary']}; margin-bottom: 0.5rem;">Model is Trained</h4>
                            <p style="color: {theme_config['text_secondary']};">Ready for face recognition</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif dataset_info['total_people'] > 0:
                    st.markdown(f"""
                    <div class="face-card">
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 3rem; color: {theme_config['warning_color']}; margin-bottom: 1rem;">⚠️</div>
                            <h4 style="color: {theme_config['text_primary']}; margin-bottom: 0.5rem;">Model Needs Training</h4>
                            <p style="color: {theme_config['text_secondary']};">You have {dataset_info['total_people']} people with {dataset_info['total_samples']} samples ready for training</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="face-card">
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 3rem; color: {theme_config['text_muted']}; margin-bottom: 1rem;">📭</div>
                            <h4 style="color: {theme_config['text_primary']}; margin-bottom: 0.5rem;">No Training Data</h4>
                            <p style="color: {theme_config['text_secondary']};">Add people to the dataset first before training</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Training Button
                if st.button("🚀 Train Model", type="primary", use_container_width=True, key="train_model_btn"):
                    if dataset_info['total_people'] > 0:
                        with st.spinner("Training face recognition model..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.text("Preparing training data...")
                            progress_bar.progress(25)

                            results = face_engine.train_model()

                            progress_bar.progress(75)
                            status_text.text("Finalizing model...")

                            progress_bar.progress(100)
                            status_text.text("Training complete!")

                            if results['status'] == 'completed':
                                # Update face training status for all trained employees
                                if st.session_state.attendance_manager:
                                    all_employees = st.session_state.attendance_manager.get_all_employees()
                                    for employee in all_employees:
                                        st.session_state.attendance_manager.update_employee_face_status(
                                            employee['employee_id'], True
                                        )

                                st.success(f"✅ Model trained successfully!")
                                st.info(f"📊 **{results['people_trained']} people** trained with **{results['total_samples']} samples**")
                                st.info("🎯 All employees are now ready for attendance tracking!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ No dataset available. Add people first.")

                # Retrain Option
                if face_engine.is_trained:
                    st.markdown("---")
                    if st.button("🔄 Retrain Model", help="Retrain the model with current dataset", key="retrain_model_btn"):
                        with st.spinner("Retraining face recognition model..."):
                            results = face_engine.train_model()

                            if results['status'] == 'completed':
                                # Update face training status for all trained employees
                                if st.session_state.attendance_manager:
                                    all_employees = st.session_state.attendance_manager.get_all_employees()
                                    for employee in all_employees:
                                        st.session_state.attendance_manager.update_employee_face_status(
                                            employee['employee_id'], True
                                        )

                                st.success(f"✅ Model retrained successfully!")
                                st.info(f"📊 **{results['people_trained']} people** trained with **{results['total_samples']} samples**")
                                st.info("🎯 All employees are now ready for attendance tracking!")
                            else:
                                st.error(f"❌ Retraining failed: {results.get('error', 'Unknown error')}")

            with col2:
                # Training Information
                st.markdown(f"""
                <div class="face-card">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📋 Training Info</h4>
                    <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem; line-height: 1.6;">
                        <p><strong>Algorithm:</strong> LBPH (Local Binary Pattern Histogram)</p>
                        <p><strong>Recommended:</strong> 50+ samples per person</p>
                        <p><strong>Training Time:</strong> ~1-2 seconds per person</p>
                        <p><strong>Model Size:</strong> Lightweight and fast</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Recognition Settings
                if face_engine.is_trained:
                    st.markdown(f"""
                    <div class="face-card">
                        <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">⚙️ Recognition Settings</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    new_threshold = st.slider(
                        "Confidence Threshold",
                        0, 100,
                        face_engine.confidence_threshold,
                        5,
                        help="Higher values = more strict recognition",
                        key="confidence_threshold_slider"
                    )

                    if new_threshold != face_engine.confidence_threshold:
                        face_engine.update_confidence_threshold(new_threshold)
                        st.success(f"✅ Threshold updated to {new_threshold}%")

        with face_tab4:
            # Test Recognition Section
            st.markdown(f"""
            <div class="face-card">
                <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">🧪 Test Recognition</h3>
            </div>
            """, unsafe_allow_html=True)

            if not face_engine.is_trained:
                st.warning("⚠️ Please train the model first before testing recognition.")
            else:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(f"""
                    <div class="face-card">
                        <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📤 Upload Test Image</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    uploaded_file = st.file_uploader(
                        "Choose an image file",
                        type=['jpg', 'jpeg', 'png'],
                        help="Upload an image to test face recognition",
                        key="test_image_upload"
                    )

                    if uploaded_file is not None:
                        # Display uploaded image
                        image = Image.open(uploaded_file)

                        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        st.image(image, caption="Uploaded Image", use_container_width=True)

                        if st.button("🔍 Test Recognition", type="primary", use_container_width=True, key="test_recognition_btn"):
                            with st.spinner("Testing face recognition..."):
                                # Convert PIL image to numpy array
                                try:
                                    img_array = np.array(image)
                                except NameError:
                                    import numpy as np
                                    img_array = np.array(image)

                                # Perform face recognition
                                face_results = face_engine.recognize_faces(img_array)

                                if face_results:
                                    st.success(f"✅ Found {len(face_results)} face(s)")

                                    for i, face in enumerate(face_results):
                                        if face['recognized_person']:
                                            st.info(f"👤 **{face['recognized_person']}** (Confidence: {face['recognition_confidence']}%)")
                                        else:
                                            st.warning(f"❓ Unknown person (Confidence: {face['recognition_confidence']}%)")
                                else:
                                    st.warning("❓ No faces detected in the image")

                with col2:
                    st.markdown(f"""
                    <div class="face-card">
                        <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">💡 Testing Tips</h4>
                        <ul style="color: {theme_config['text_secondary']}; font-size: 0.9rem; line-height: 1.6;">
                            <li>Use clear, well-lit images</li>
                            <li>Face should be clearly visible</li>
                            <li>Avoid heavy shadows or reflections</li>
                            <li>Test with different angles and expressions</li>
                            <li>Higher confidence = better match</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Recognition Statistics
                    if face_engine.is_trained:
                        st.markdown(f"""
                        <div class="face-card">
                            <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📊 Model Statistics</h4>
                            <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                                <p><strong>Trained People:</strong> {dataset_info['total_people']}</p>
                                <p><strong>Total Samples:</strong> {dataset_info['total_samples']}</p>
                                <p><strong>Confidence Threshold:</strong> {face_engine.confidence_threshold}%</p>
                                <p><strong>Model Status:</strong> <span style="color: {theme_config['success_color']};">Ready</span></p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    with tab7:
        # Live Attendance Tab
        current_theme = theme_manager.get_current_theme()
        theme_config = theme_manager.get_theme_config(current_theme)

        # Header with modern styling
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {theme_config['accent_color']}15 0%, {theme_config['success_color']}15 100%);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 8px 32px {theme_config['shadow']};
        ">
            <h1 style="
                margin: 0 0 1rem 0;
                font-size: 2.5rem;
                background: linear-gradient(135deg, {theme_config['accent_color']}, {theme_config['success_color']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">📊 Live Attendance Dashboard</h1>
            <p style="
                color: {theme_config['text_secondary']};
                margin: 0;
                font-size: 1.1rem;
            ">Real-time attendance tracking and management</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.attendance_manager:
            st.error("❌ Attendance system is not available. Please check your installation.")
            st.stop()

        # Get attendance data
        attendance_manager = st.session_state.attendance_manager
        today_attendance = attendance_manager.get_today_attendance()
        attendance_stats = attendance_manager.get_attendance_stats()

        # Real-time notifications area
        notification_container = st.container()

        # Check for real-time attendance updates
        attendance_updated = False
        if hasattr(st.session_state, 'webcam_detector') and st.session_state.webcam_detector:
            attendance_summary = st.session_state.webcam_detector.get_attendance_summary()
            attendance_updated = attendance_summary.get('attendance_updated', False)

            if attendance_updated:
                # Get the latest detection
                recent_detections = attendance_summary.get('recent_detections', [])
                if recent_detections:
                    latest_detection = recent_detections[-1]
                    with notification_container:
                        st.success(f"🎉 **New Attendance Recorded!** {latest_detection['name']} ({latest_detection['confidence']:.0f}% confidence) at {datetime.fromtimestamp(latest_detection['timestamp']).strftime('%H:%M:%S')}")

                # Reset the flag and refresh data
                st.session_state.webcam_detector.attendance_updated = False
                today_attendance = attendance_manager.get_today_attendance()
                attendance_stats = attendance_manager.get_attendance_stats()
                time.sleep(1)  # Brief pause to show the notification
                st.rerun()

        # Live detection status indicator
        if hasattr(st.session_state, 'webcam_detector') and st.session_state.webcam_detector:
            latest_stats = st.session_state.webcam_detector.get_latest_stats()
            if latest_stats and latest_stats.get('frame_count', 0) > 0:
                st.info(f"📹 **Live Camera Active** - Processing frames in real-time | Last update: {time.strftime('%H:%M:%S')}")
            else:
                st.warning("📹 **Camera Inactive** - Start the camera in the Live Detection tab to enable real-time attendance tracking")
        else:
            st.info("📹 **Camera Status** - Go to Live Detection tab to start real-time face recognition")

        # Create attendance dashboard layout
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="
                background: {theme_config['card_bg']};
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                border: 1px solid {theme_config['border_color']};
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="font-size: 2rem; color: {theme_config['success_color']}; margin-bottom: 0.5rem;">
                    {attendance_stats['present_count']}
                </div>
                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                    Present Today
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="
                background: {theme_config['card_bg']};
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                border: 1px solid {theme_config['border_color']};
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="font-size: 2rem; color: {theme_config['warning_color']}; margin-bottom: 0.5rem;">
                    {attendance_stats['absent_count']}
                </div>
                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                    Absent Today
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="
                background: {theme_config['card_bg']};
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                border: 1px solid {theme_config['border_color']};
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="font-size: 2rem; color: {theme_config['info_color']}; margin-bottom: 0.5rem;">
                    {attendance_stats['attendance_rate']:.1f}%
                </div>
                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                    Attendance Rate
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="
                background: {theme_config['card_bg']};
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                border: 1px solid {theme_config['border_color']};
                box-shadow: 0 4px 20px {theme_config['shadow']};
            ">
                <div style="font-size: 2rem; color: {theme_config['accent_color']}; margin-bottom: 0.5rem;">
                    {attendance_stats['total_employees']}
                </div>
                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                    Total Employees
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Employee Database Viewer Section
        st.markdown(f"""
        <div style="
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 2rem;
        ">
            <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">👥 Employee Database</h3>
        </div>
        """, unsafe_allow_html=True)

        # Employee Database Controls
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            if st.button("👥 View All Employees", type="primary", use_container_width=True):
                st.session_state.show_employee_database = True

        with col2:
            if st.button("📊 Employee Statistics", type="secondary", use_container_width=True):
                st.session_state.show_employee_stats = True

        with col3:
            if st.button("🔄 Refresh", help="Refresh all data", use_container_width=True):
                st.rerun()

        with col4:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto", value=st.session_state.get('auto_refresh_attendance', False),
                                     help="Auto-refresh every 5 seconds")
            st.session_state.auto_refresh_attendance = auto_refresh

        # Auto-refresh functionality
        if st.session_state.get('auto_refresh_attendance', False):
            # Add auto-refresh timer
            if 'last_refresh_time' not in st.session_state:
                st.session_state.last_refresh_time = time.time()

            current_time = time.time()
            if current_time - st.session_state.last_refresh_time > 5:  # Refresh every 5 seconds
                st.session_state.last_refresh_time = current_time
                st.rerun()

        # Employee Database Display
        if st.session_state.get('show_employee_database', False):
            all_employees = attendance_manager.get_all_employees()

            if all_employees:
                st.markdown(f"""
                <div style="
                    background: {theme_config['secondary_bg']};
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border: 1px solid {theme_config['border_color']};
                ">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📋 Complete Employee Database</h4>
                </div>
                """, unsafe_allow_html=True)

                # Convert to DataFrame for better display
                import pandas as pd
                df_employees = pd.DataFrame(all_employees)

                # Add attendance statistics for each employee
                for idx, employee in enumerate(all_employees):
                    # Get last 30 days attendance
                    thirty_days_ago = date.today() - timedelta(days=30)
                    employee_attendance = [
                        att for att in attendance_manager.get_attendance_by_date_range(thirty_days_ago, date.today())
                        if att['employee_id'] == employee['employee_id']
                    ]

                    attendance_count = len(employee_attendance)
                    attendance_rate = (attendance_count / 30) * 100

                    # Add to dataframe
                    df_employees.loc[idx, 'attendance_30_days'] = attendance_count
                    df_employees.loc[idx, 'attendance_rate_30d'] = f"{attendance_rate:.1f}%"

                # Reorder columns for better display
                column_order = ['employee_id', 'name', 'department', 'face_trained', 'attendance_30_days', 'attendance_rate_30d', 'created_date']
                df_employees = df_employees.reindex(columns=[col for col in column_order if col in df_employees.columns])

                # Rename columns for better display
                df_employees.columns = ['Employee ID', 'Name', 'Department', 'Face Trained', '30-Day Attendance', '30-Day Rate', 'Created Date']

                st.dataframe(
                    df_employees,
                    use_container_width=True,
                    hide_index=True
                )

                # Summary statistics
                total_employees = len(all_employees)
                trained_employees = sum(1 for emp in all_employees if emp.get('face_trained', False))
                avg_attendance_rate = df_employees['30-Day Rate'].str.rstrip('%').astype(float).mean()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Employees", total_employees)
                with col2:
                    st.metric("Face Trained", f"{trained_employees}/{total_employees}")
                with col3:
                    st.metric("Avg 30-Day Rate", f"{avg_attendance_rate:.1f}%")

                if st.button("❌ Close Employee Database", use_container_width=True):
                    st.session_state.show_employee_database = False
                    st.rerun()
            else:
                st.info("👥 No employees registered yet. Please add employees in the Face Recognition tab first.")

        # Employee Statistics Display
        if st.session_state.get('show_employee_stats', False):
            all_employees = attendance_manager.get_all_employees()

            if all_employees:
                st.markdown(f"""
                <div style="
                    background: {theme_config['secondary_bg']};
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border: 1px solid {theme_config['border_color']};
                ">
                    <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📊 Employee Statistics Dashboard</h4>
                </div>
                """, unsafe_allow_html=True)

                # Department-wise statistics
                departments = {}
                for employee in all_employees:
                    dept = employee.get('department', 'Not Specified')
                    if dept not in departments:
                        departments[dept] = {'total': 0, 'trained': 0, 'attendance_sum': 0}
                    departments[dept]['total'] += 1
                    if employee.get('face_trained', False):
                        departments[dept]['trained'] += 1

                    # Get attendance for this employee
                    thirty_days_ago = date.today() - timedelta(days=30)
                    employee_attendance = [
                        att for att in attendance_manager.get_attendance_by_date_range(thirty_days_ago, date.today())
                        if att['employee_id'] == employee['employee_id']
                    ]
                    departments[dept]['attendance_sum'] += len(employee_attendance)

                # Display department statistics
                st.markdown("**📈 Department-wise Statistics:**")

                for dept, stats in departments.items():
                    avg_attendance = (stats['attendance_sum'] / stats['total']) if stats['total'] > 0 else 0
                    attendance_rate = (avg_attendance / 30) * 100
                    training_rate = (stats['trained'] / stats['total']) * 100 if stats['total'] > 0 else 0

                    # Color coding
                    if attendance_rate >= 80:
                        rate_color = theme_config['success_color']
                        rate_icon = "🟢"
                    elif attendance_rate >= 60:
                        rate_color = theme_config['warning_color']
                        rate_icon = "🟡"
                    else:
                        rate_color = theme_config['danger_color']
                        rate_icon = "🔴"

                    st.markdown(f"""
                    <div style="
                        background: {theme_config['card_bg']};
                        border-radius: 10px;
                        padding: 1rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid {rate_color};
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: {theme_config['text_primary']};">🏢 {dept}</strong>
                                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem; margin-top: 0.3rem;">
                                    👥 {stats['total']} employees | 🎯 {stats['trained']} trained ({training_rate:.0f}%)
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: {rate_color}; font-weight: bold; font-size: 1.1rem;">
                                    {rate_icon} {attendance_rate:.1f}%
                                </div>
                                <div style="color: {theme_config['text_secondary']}; font-size: 0.8rem;">
                                    Avg Attendance
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("❌ Close Statistics", use_container_width=True):
                    st.session_state.show_employee_stats = False
                    st.rerun()
            else:
                st.info("👥 No employees registered yet. Please add employees in the Face Recognition tab first.")

        st.markdown("---")

        # Real-time attendance table
        st.markdown(f"""
        <div style="
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 2rem;
        ">
            <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📋 Today's Attendance</h3>
        </div>
        """, unsafe_allow_html=True)

        if today_attendance:
            # Convert to DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(today_attendance)

            # Format time column
            if 'time_in' in df.columns:
                df['time_in'] = pd.to_datetime(df['time_in']).dt.strftime('%H:%M:%S')

            # Reorder columns
            column_order = ['employee_id', 'name', 'department', 'time_in', 'status', 'camera_location']
            df = df.reindex(columns=[col for col in column_order if col in df.columns])

            # Rename columns for better display
            df.columns = ['Employee ID', 'Name', 'Department', 'Time In', 'Status', 'Camera Location']

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("📭 No attendance records for today yet.")

        # Export functionality
        st.markdown("---")
        st.markdown(f"""
        <div style="
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 2rem;
        ">
            <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📥 Export Attendance Data</h3>
        </div>
        """, unsafe_allow_html=True)

        # Date range selection for export
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today(), key="attendance_start_date")
        with col2:
            end_date = st.date_input("End Date", value=date.today(), key="attendance_end_date")

        if start_date > end_date:
            st.error("❌ Start date must be before or equal to end date")
        else:
            # Export buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Export as CSV", type="primary", use_container_width=True):
                    csv_data = attendance_manager.export_attendance_to_csv(start_date, end_date)

                    if csv_data and not csv_data.startswith("No attendance data") and not csv_data.startswith("Error"):
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"attendance_{start_date}_{end_date}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.success("✅ CSV export ready for download!")
                    else:
                        st.warning(csv_data if csv_data else "No data available for export")

            with col2:
                if st.button("📈 Export as Excel", type="secondary", use_container_width=True):
                    try:
                        excel_data = attendance_manager.export_attendance_to_excel(start_date, end_date)

                        if excel_data:
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"attendance_{start_date}_{end_date}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.success("✅ Excel export ready for download!")
                        else:
                            st.warning("No data available for Excel export")
                    except Exception as e:
                        st.error(f"❌ Excel export failed: {str(e)}")

            with col3:
                if st.button("🔄 Reset Today's Attendance", help="Clear all attendance records for today", use_container_width=True):
                    if st.session_state.get('confirm_reset_attendance', False):
                        if attendance_manager.reset_daily_attendance():
                            st.success("✅ Today's attendance has been reset")
                            st.session_state['confirm_reset_attendance'] = False
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset attendance")
                    else:
                        st.session_state['confirm_reset_attendance'] = True
                        st.warning("⚠️ Click again to confirm reset")

        # Attendance Visualization Dashboard
        st.markdown("---")
        st.markdown(f"""
        <div style="
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 2rem;
        ">
            <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📊 Attendance Analytics & Visualizations</h3>
        </div>
        """, unsafe_allow_html=True)

        # Visualization Controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Today's Overview", type="primary", use_container_width=True):
                st.session_state.show_today_viz = True

        with col2:
            if st.button("📈 Department Analysis", type="secondary", use_container_width=True):
                st.session_state.show_dept_viz = True

        with col3:
            if st.button("📅 Weekly Trends", help="Show weekly attendance trends", use_container_width=True):
                st.session_state.show_weekly_viz = True

        # Today's Attendance Visualization
        if st.session_state.get('show_today_viz', False):
            st.markdown(f"""
            <div style="
                background: {theme_config['secondary_bg']};
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid {theme_config['border_color']};
            ">
                <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📊 Today's Attendance Overview</h4>
            </div>
            """, unsafe_allow_html=True)

            # Create pie chart for today's attendance
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Get today's data
                present_count = attendance_stats['present_count']
                absent_count = attendance_stats['absent_count']
                total_employees = attendance_stats['total_employees']

                if total_employees > 0:
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=['Present', 'Absent'],
                        values=[present_count, absent_count],
                        hole=0.4,
                        marker_colors=[theme_config['success_color'], theme_config['danger_color']]
                    )])

                    fig.update_layout(
                        title="Today's Attendance Distribution",
                        font=dict(size=14),
                        showlegend=True,
                        height=400,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Attendance Rate", f"{attendance_stats['attendance_rate']:.1f}%")
                    with col2:
                        st.metric("Present", present_count)
                    with col3:
                        st.metric("Absent", absent_count)
                else:
                    st.info("📭 No employee data available for visualization")

            except ImportError:
                st.warning("📊 Install plotly for advanced visualizations: `pip install plotly`")

                # Fallback simple visualization
                if attendance_stats['total_employees'] > 0:
                    attendance_rate = attendance_stats['attendance_rate']
                    st.progress(attendance_rate / 100, text=f"Attendance Rate: {attendance_rate:.1f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Present Today", attendance_stats['present_count'])
                    with col2:
                        st.metric("Absent Today", attendance_stats['absent_count'])

            if st.button("❌ Close Today's Overview", use_container_width=True):
                st.session_state.show_today_viz = False
                st.rerun()

        # Department Analysis Visualization
        if st.session_state.get('show_dept_viz', False):
            st.markdown(f"""
            <div style="
                background: {theme_config['secondary_bg']};
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid {theme_config['border_color']};
            ">
                <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📈 Department-wise Analysis</h4>
            </div>
            """, unsafe_allow_html=True)

            all_employees = attendance_manager.get_all_employees()

            if all_employees:
                # Calculate department statistics
                departments = {}
                for employee in all_employees:
                    dept = employee.get('department', 'Not Specified')
                    if dept not in departments:
                        departments[dept] = {'total': 0, 'present': 0}
                    departments[dept]['total'] += 1

                    # Check if present today
                    employee_present = any(
                        att['employee_id'] == employee['employee_id']
                        for att in today_attendance
                    )
                    if employee_present:
                        departments[dept]['present'] += 1

                try:
                    import plotly.express as px
                    import plotly.graph_objects as go

                    # Create department comparison chart
                    dept_names = list(departments.keys())
                    dept_totals = [departments[dept]['total'] for dept in dept_names]
                    dept_present = [departments[dept]['present'] for dept in dept_names]
                    dept_rates = [
                        (departments[dept]['present'] / departments[dept]['total']) * 100
                        if departments[dept]['total'] > 0 else 0
                        for dept in dept_names
                    ]

                    # Bar chart for department attendance rates
                    fig = go.Figure(data=[
                        go.Bar(
                            x=dept_names,
                            y=dept_rates,
                            marker_color=[
                                theme_config['success_color'] if rate >= 80 else
                                theme_config['warning_color'] if rate >= 60 else
                                theme_config['danger_color']
                                for rate in dept_rates
                            ],
                            text=[f"{rate:.1f}%" for rate in dept_rates],
                            textposition='auto'
                        )
                    ])

                    fig.update_layout(
                        title="Department-wise Attendance Rates",
                        xaxis_title="Department",
                        yaxis_title="Attendance Rate (%)",
                        font=dict(size=12),
                        height=400,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Department details table
                    import pandas as pd
                    dept_df = pd.DataFrame([
                        {
                            'Department': dept,
                            'Total Employees': departments[dept]['total'],
                            'Present Today': departments[dept]['present'],
                            'Attendance Rate': f"{(departments[dept]['present'] / departments[dept]['total']) * 100:.1f}%"
                        }
                        for dept in dept_names
                    ])

                    st.dataframe(dept_df, use_container_width=True, hide_index=True)

                except ImportError:
                    # Fallback without plotly
                    st.markdown("**Department Statistics:**")
                    for dept, stats in departments.items():
                        rate = (stats['present'] / stats['total']) * 100 if stats['total'] > 0 else 0
                        st.metric(
                            f"🏢 {dept}",
                            f"{rate:.1f}%",
                            f"{stats['present']}/{stats['total']} present"
                        )
            else:
                st.info("👥 No employees registered yet.")

            if st.button("❌ Close Department Analysis", use_container_width=True):
                st.session_state.show_dept_viz = False
                st.rerun()

        # Weekly Trends Visualization
        if st.session_state.get('show_weekly_viz', False):
            st.markdown(f"""
            <div style="
                background: {theme_config['secondary_bg']};
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid {theme_config['border_color']};
            ">
                <h4 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">📅 Weekly Attendance Trends</h4>
            </div>
            """, unsafe_allow_html=True)

            # Get last 7 days attendance data
            weekly_data = []
            for i in range(7):
                check_date = date.today() - timedelta(days=i)
                day_attendance = attendance_manager.get_attendance_by_date_range(check_date, check_date)
                weekly_data.append({
                    'date': check_date,
                    'day': check_date.strftime('%A'),
                    'count': len(day_attendance),
                    'rate': (len(day_attendance) / attendance_stats['total_employees']) * 100 if attendance_stats['total_employees'] > 0 else 0
                })

            weekly_data.reverse()  # Show oldest to newest

            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Line chart for weekly trends
                dates = [data['day'] for data in weekly_data]
                counts = [data['count'] for data in weekly_data]
                rates = [data['rate'] for data in weekly_data]

                fig = go.Figure()

                # Add attendance count line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines+markers',
                    name='Attendance Count',
                    line=dict(color=theme_config['accent_color'], width=3),
                    marker=dict(size=8)
                ))

                fig.update_layout(
                    title="7-Day Attendance Trend",
                    xaxis_title="Day",
                    yaxis_title="Number of Attendees",
                    font=dict(size=12),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=50)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Weekly summary
                avg_attendance = sum(counts) / len(counts) if counts else 0
                avg_rate = sum(rates) / len(rates) if rates else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Weekly Average", f"{avg_attendance:.1f}")
                with col2:
                    st.metric("Average Rate", f"{avg_rate:.1f}%")
                with col3:
                    best_day = max(weekly_data, key=lambda x: x['count'])
                    st.metric("Best Day", f"{best_day['day']} ({best_day['count']})")

            except ImportError:
                # Fallback without plotly
                st.markdown("**Weekly Summary:**")
                for data in weekly_data:
                    st.metric(
                        data['day'],
                        data['count'],
                        f"{data['rate']:.1f}% rate"
                    )

            if st.button("❌ Close Weekly Trends", use_container_width=True):
                st.session_state.show_weekly_viz = False
                st.rerun()

        # Recent detections from live camera
        if hasattr(st.session_state, 'webcam_detector') and st.session_state.webcam_detector:
            attendance_summary = st.session_state.webcam_detector.get_attendance_summary()
            recent_detections = attendance_summary.get('recent_detections', [])

            if recent_detections:
                st.markdown("---")
                st.markdown(f"""
                <div style="
                    background: {theme_config['card_bg']};
                    border-radius: 15px;
                    padding: 1.5rem;
                    border: 1px solid {theme_config['border_color']};
                    box-shadow: 0 4px 20px {theme_config['shadow']};
                    margin-bottom: 2rem;
                ">
                    <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">🎥 Recent Live Detections</h3>
                </div>
                """, unsafe_allow_html=True)

                for detection in recent_detections[-5:]:  # Show last 5 detections
                    detection_time = datetime.fromtimestamp(detection['timestamp']).strftime('%H:%M:%S')
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {theme_config['success_color']}10 0%, {theme_config['card_bg']} 100%);
                        border-radius: 10px;
                        padding: 1rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid {theme_config['success_color']};
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: {theme_config['text_primary']};">👤 {detection['name']}</strong>
                                <span style="color: {theme_config['text_secondary']}; margin-left: 1rem;">
                                    🆔 {detection['employee_id']} | 🏢 {detection['department']}
                                </span>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: {theme_config['success_color']}; font-weight: bold;">✓ {detection['status']}</div>
                                <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                                    🕒 {detection_time} | 🎯 {detection['confidence']:.0f}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Manual attendance management
        st.markdown("---")
        st.markdown(f"""
        <div style="
            background: {theme_config['card_bg']};
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid {theme_config['border_color']};
            box-shadow: 0 4px 20px {theme_config['shadow']};
            margin-bottom: 2rem;
        ">
            <h3 style="color: {theme_config['text_primary']}; margin: 0 0 1rem 0;">⚙️ Manual Attendance Management</h3>
        </div>
        """, unsafe_allow_html=True)

        # Get all employees for manual management
        all_employees = attendance_manager.get_all_employees()

        if all_employees:
            # Manual attendance marking
            with st.expander("➕ Mark Manual Attendance", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    employee_names = [emp['name'] for emp in all_employees]
                    selected_employee = st.selectbox("Select Employee", employee_names, key="manual_attendance_employee")

                with col2:
                    manual_date = st.date_input("Date", value=date.today(), key="manual_attendance_date")

                if st.button("✅ Mark Present", type="primary", key="mark_manual_attendance"):
                    selected_emp = next(emp for emp in all_employees if emp['name'] == selected_employee)

                    # Check if attendance already exists
                    existing_attendance = attendance_manager.get_attendance_by_date_range(manual_date, manual_date)
                    employee_already_marked = any(att['employee_id'] == selected_emp['employee_id'] for att in existing_attendance)

                    if employee_already_marked:
                        st.warning(f"⚠️ {selected_employee} is already marked present for {manual_date}")
                    else:
                        # Record manual attendance
                        success = attendance_manager.record_attendance(
                            selected_emp['employee_id'],
                            100.0,  # Manual entry gets 100% confidence
                            "Manual Entry",
                            "manual_session"
                        )

                        if success:
                            st.success(f"✅ Successfully marked {selected_employee} as present for {manual_date}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to record manual attendance")

            # Attendance history view
            with st.expander("📊 View Attendance History", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    history_start = st.date_input("From Date", value=date.today() - timedelta(days=7), key="history_start")
                with col2:
                    history_end = st.date_input("To Date", value=date.today(), key="history_end")

                if st.button("📋 Load History", key="load_attendance_history"):
                    if history_start <= history_end:
                        history_data = attendance_manager.get_attendance_by_date_range(history_start, history_end)

                        if history_data:
                            import pandas as pd
                            df_history = pd.DataFrame(history_data)

                            # Format time column
                            if 'time_in' in df_history.columns:
                                df_history['time_in'] = pd.to_datetime(df_history['time_in']).dt.strftime('%H:%M:%S')

                            # Reorder columns
                            column_order = ['date', 'employee_id', 'name', 'department', 'time_in', 'status', 'camera_location']
                            df_history = df_history.reindex(columns=[col for col in column_order if col in df_history.columns])

                            # Rename columns for better display
                            df_history.columns = ['Date', 'Employee ID', 'Name', 'Department', 'Time In', 'Status', 'Camera Location']

                            st.dataframe(
                                df_history,
                                use_container_width=True,
                                hide_index=True
                            )

                            # Summary statistics for the period
                            unique_dates = df_history['Date'].nunique()
                            unique_employees = df_history['Employee ID'].nunique()
                            total_records = len(df_history)

                            st.info(f"📊 **Summary:** {total_records} attendance records for {unique_employees} employees across {unique_dates} days")
                        else:
                            st.info("📭 No attendance records found for the selected date range")
                    else:
                        st.error("❌ Start date must be before or equal to end date")

            # Employee attendance summary
            with st.expander("👥 Employee Attendance Summary", expanded=False):
                if all_employees:
                    st.markdown("**Individual Employee Statistics:**")

                    for employee in all_employees:
                        # Get attendance count for last 30 days
                        thirty_days_ago = date.today() - timedelta(days=30)
                        employee_attendance = [
                            att for att in attendance_manager.get_attendance_by_date_range(thirty_days_ago, date.today())
                            if att['employee_id'] == employee['employee_id']
                        ]

                        attendance_count = len(employee_attendance)
                        attendance_rate = (attendance_count / 30) * 100

                        # Color coding based on attendance rate
                        if attendance_rate >= 90:
                            rate_color = theme_config['success_color']
                            rate_icon = "🟢"
                        elif attendance_rate >= 70:
                            rate_color = theme_config['warning_color']
                            rate_icon = "🟡"
                        else:
                            rate_color = theme_config['danger_color']
                            rate_icon = "🔴"

                        st.markdown(f"""
                        <div style="
                            background: {theme_config['secondary_bg']};
                            border-radius: 8px;
                            padding: 1rem;
                            margin-bottom: 0.5rem;
                            border-left: 3px solid {rate_color};
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: {theme_config['text_primary']};">👤 {employee['name']}</strong>
                                    <span style="color: {theme_config['text_secondary']}; margin-left: 1rem;">
                                        🆔 {employee['employee_id']} | 🏢 {employee['department']}
                                    </span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="color: {rate_color}; font-weight: bold;">
                                        {rate_icon} {attendance_rate:.1f}%
                                    </div>
                                    <div style="color: {theme_config['text_secondary']}; font-size: 0.9rem;">
                                        {attendance_count}/30 days
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👥 No employees registered yet. Please add employees in the Face Recognition tab first.")

if __name__ == "__main__":
    main()
