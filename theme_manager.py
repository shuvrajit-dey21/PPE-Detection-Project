"""
Theme Management System for PPE Monitoring Application
Provides comprehensive dark/light theme support with proper text visibility
"""

import streamlit as st
from typing import Dict, Any

class ThemeManager:
    """Manages application themes with proper text visibility"""
    
    def __init__(self):
        self.themes = {
            "light": {
                "name": "Light Theme",
                "icon": "☀️",
                "primary_bg": "#ffffff",
                "secondary_bg": "#f8f9fa",
                "tertiary_bg": "#e9ecef",
                "card_bg": "#ffffff",
                "text_primary": "#212529",
                "text_secondary": "#6c757d",
                "text_muted": "#868e96",
                "border_color": "#dee2e6",
                "accent_color": "#007bff",
                "success_color": "#28a745",
                "warning_color": "#ffc107",
                "danger_color": "#dc3545",
                "info_color": "#17a2b8",
                "shadow": "rgba(0,0,0,0.1)",
                "shadow_hover": "rgba(0,0,0,0.15)",
                "gradient_start": "#ffffff",
                "gradient_end": "#f8f9fa"
            },
            "dark": {
                "name": "Dark Theme",
                "icon": "🌙",
                "primary_bg": "#1a1a1a",
                "secondary_bg": "#2d2d2d",
                "tertiary_bg": "#404040",
                "card_bg": "#2d2d2d",
                "text_primary": "#ffffff",
                "text_secondary": "#e0e0e0",
                "text_muted": "#b0b0b0",
                "border_color": "#404040",
                "accent_color": "#4dabf7",
                "success_color": "#51cf66",
                "warning_color": "#ffd43b",
                "danger_color": "#ff6b6b",
                "info_color": "#74c0fc",
                "shadow": "rgba(0,0,0,0.3)",
                "shadow_hover": "rgba(0,0,0,0.4)",
                "gradient_start": "#2d2d2d",
                "gradient_end": "#1a1a1a"
            }
        }
    
    def get_current_theme(self) -> str:
        """Get current theme from session state"""
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'  # Default to light theme
        return st.session_state.theme
    
    def set_theme(self, theme_name: str):
        """Set current theme"""
        if theme_name in self.themes:
            st.session_state.theme = theme_name
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        current = self.get_current_theme()
        new_theme = 'dark' if current == 'light' else 'light'
        self.set_theme(new_theme)
    
    def get_theme_config(self, theme_name: str = None) -> Dict[str, Any]:
        """Get theme configuration"""
        if theme_name is None:
            theme_name = self.get_current_theme()
        return self.themes.get(theme_name, self.themes['light'])
    
    def create_theme_toggle(self):
        """Create theme toggle button in sidebar with enhanced styling"""
        current_theme = self.get_current_theme()
        theme_config = self.get_theme_config(current_theme)

        # Create toggle button
        other_theme = 'dark' if current_theme == 'light' else 'light'
        other_config = self.get_theme_config(other_theme)

        # Enhanced styling for the theme toggle button with perfect animations
        button_style = f"""
        <style>
        /* Enhanced Theme Toggle Button */
        div[data-testid="stSidebar"] .stButton > button {{
            background: linear-gradient(135deg, {theme_config['accent_color']} 0%, {theme_config['info_color']} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            padding: 0.8rem 1.2rem !important;
            box-shadow: 0 6px 20px {theme_config['shadow']} !important;
            position: relative !important;
            overflow: hidden !important;
            animation: themeButtonGlow 0.6s ease-out !important;
        }}

        div[data-testid="stSidebar"] .stButton > button::before {{
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
            transition: left 0.5s !important;
        }}

        div[data-testid="stSidebar"] .stButton > button:hover {{
            background: linear-gradient(135deg, {theme_config['info_color']} 0%, {theme_config['accent_color']} 100%) !important;
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 10px 30px {theme_config['shadow_hover']} !important;
        }}

        div[data-testid="stSidebar"] .stButton > button:hover::before {{
            left: 100% !important;
        }}

        div[data-testid="stSidebar"] .stButton > button:active {{
            transform: translateY(-1px) scale(0.98) !important;
            box-shadow: 0 4px 15px {theme_config['shadow']} !important;
        }}

        @keyframes themeButtonGlow {{
            0% {{
                opacity: 0;
                transform: scale(0.9) translateY(10px);
                box-shadow: 0 0 0 {theme_config['shadow']};
            }}
            50% {{
                opacity: 1;
                transform: scale(1.02);
                box-shadow: 0 8px 25px {theme_config['shadow_hover']};
            }}
            100% {{
                opacity: 1;
                transform: scale(1) translateY(0);
                box-shadow: 0 6px 20px {theme_config['shadow']};
            }}
        }}
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        if st.sidebar.button(
            f"{other_config['icon']} Switch to {other_config['name']}",
            use_container_width=True,
            help=f"Switch from {theme_config['name']} to {other_config['name']}"
        ):
            self.toggle_theme()
            st.rerun()
    
    def get_css_variables(self, theme_name: str = None) -> str:
        """Generate CSS variables for the theme"""
        config = self.get_theme_config(theme_name)
        
        css_vars = ":root {\n"
        for key, value in config.items():
            if key not in ['name', 'icon']:
                css_var_name = f"--{key.replace('_', '-')}"
                css_vars += f"    {css_var_name}: {value};\n"
        css_vars += "}\n"
        
        return css_vars
    
    def apply_theme_css(self):
        """Apply comprehensive theme CSS"""
        current_theme = self.get_current_theme()
        css_variables = self.get_css_variables(current_theme)
        
        # Comprehensive theme CSS with enhanced dark theme support
        theme_css = f"""
        <style>
        {css_variables}

        /* Global theme application */
        .stApp {{
            background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
            color: var(--text-primary) !important;
            min-height: 100vh;
        }}

        /* Enhanced Sidebar theming with higher specificity */
        .css-1d391kg, .css-1cypcdb, .css-17eq0hr, .css-1544g2n, .css-1y4p8pa,
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
            background: var(--secondary-bg) !important;
            border-right: 1px solid var(--border-color) !important;
        }}

        /* Sidebar content styling with enhanced targeting */
        .css-1d391kg .stMarkdown, .css-1cypcdb .stMarkdown, .css-1544g2n .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown {{
            color: var(--text-primary) !important;
        }}

        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6,
        .css-1cypcdb h1, .css-1cypcdb h2, .css-1cypcdb h3, .css-1cypcdb h4, .css-1cypcdb h5, .css-1cypcdb h6,
        .css-1544g2n h1, .css-1544g2n h2, .css-1544g2n h3, .css-1544g2n h4, .css-1544g2n h5, .css-1544g2n h6,
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5, section[data-testid="stSidebar"] h6 {{
            color: var(--text-primary) !important;
        }}

        /* Sidebar labels and text with enhanced specificity */
        .css-1d391kg label, .css-1cypcdb label, .css-1544g2n label,
        section[data-testid="stSidebar"] label {{
            color: var(--text-primary) !important;
        }}

        .css-1d391kg .stSelectbox label, .css-1cypcdb .stSelectbox label, .css-1544g2n .stSelectbox label,
        .css-1d391kg .stSlider label, .css-1cypcdb .stSlider label, .css-1544g2n .stSlider label,
        .css-1d391kg .stCheckbox label, .css-1cypcdb .stCheckbox label, .css-1544g2n .stCheckbox label,
        section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stCheckbox label {{
            color: var(--text-primary) !important;
        }}

        /* Enhanced sidebar button styling */
        .css-1d391kg .stButton > button, .css-1cypcdb .stButton > button, .css-1544g2n .stButton > button,
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] button[kind="secondary"] {{
            background: var(--accent-color) !important;
            color: white !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}

        .css-1d391kg .stButton > button:hover, .css-1cypcdb .stButton > button:hover, .css-1544g2n .stButton > button:hover,
        section[data-testid="stSidebar"] .stButton > button:hover,
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background: var(--accent-color) !important;
            opacity: 0.9 !important;
            box-shadow: 0 4px 12px var(--shadow-hover) !important;
            transform: translateY(-1px) !important;
        }}

        /* Sidebar input fields with enhanced targeting */
        .css-1d391kg .stSelectbox select, .css-1cypcdb .stSelectbox select, .css-1544g2n .stSelectbox select,
        section[data-testid="stSidebar"] .stSelectbox select {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .css-1d391kg .stNumberInput input, .css-1cypcdb .stNumberInput input, .css-1544g2n .stNumberInput input {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        /* Main content area */
        .main .block-container {{
            background: transparent;
            color: var(--text-primary) !important;
            padding-top: 2rem;
        }}

        /* Text elements - Enhanced for dark theme */
        .stMarkdown, .stText, p, span, div, label {{
            color: var(--text-primary) !important;
        }}

        .stMarkdown p, .stMarkdown span, .stMarkdown div {{
            color: var(--text-primary) !important;
        }}

        /* Headers with better contrast */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            text-shadow: 0 1px 2px var(--shadow);
        }}

        /* Metric components with theme support */
        .metric-container, [data-testid="metric-container"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 8px var(--shadow);
        }}

        [data-testid="metric-container"] > div {{
            color: var(--text-primary) !important;
        }}

        [data-testid="metric-container"] label {{
            color: var(--text-secondary) !important;
        }}

        /* Cards and containers */
        .stContainer, .element-container {{
            color: var(--text-primary) !important;
        }}

        /* Enhanced button styling */
        .stButton > button {{
            background: var(--accent-color) !important;
            color: white !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background: var(--accent-color) !important;
            opacity: 0.9;
            box-shadow: 0 4px 12px var(--shadow-hover) !important;
            transform: translateY(-1px);
        }}

        .stButton > button:focus {{
            box-shadow: 0 0 0 2px var(--accent-color) !important;
        }}

        /* Input fields with dark theme support */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px;
        }}

        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 2px rgba(var(--accent-color), 0.2) !important;
        }}

        /* Checkboxes and radio buttons */
        .stCheckbox > label, .stRadio > label {{
            color: var(--text-primary) !important;
        }}

        .stCheckbox input[type="checkbox"] {{
            accent-color: var(--accent-color);
        }}

        /* Expanders with enhanced theming */
        .streamlit-expanderHeader {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px 8px 0 0;
        }}

        .streamlit-expanderContent {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px;
            color: var(--text-primary) !important;
        }}

        /* Tabs with better dark theme support */
        .stTabs [data-baseweb="tab-list"] {{
            background: var(--secondary-bg) !important;
            border-bottom: 1px solid var(--border-color) !important;
            border-radius: 8px 8px 0 0;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: var(--text-secondary) !important;
            background: transparent !important;
            border-radius: 6px 6px 0 0;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--accent-color) !important;
            background: var(--card-bg) !important;
            border-bottom: 2px solid var(--accent-color) !important;
        }}

        .stTabs [data-baseweb="tab-panel"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px;
            padding: 1rem;
        }}

        /* Progress bars */
        .stProgress > div > div > div {{
            background: var(--accent-color) !important;
        }}

        .stProgress > div > div {{
            background: var(--tertiary-bg) !important;
        }}

        /* Enhanced alert styling */
        .stAlert {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px;
        }}

        /* Success alerts */
        .stSuccess {{
            background: var(--success-color) !important;
            color: white !important;
            border-color: var(--success-color) !important;
        }}

        /* Warning alerts */
        .stWarning {{
            background: var(--warning-color) !important;
            color: var(--text-primary) !important;
            border-color: var(--warning-color) !important;
        }}

        /* Error alerts */
        .stError {{
            background: var(--danger-color) !important;
            color: white !important;
            border-color: var(--danger-color) !important;
        }}

        /* Info alerts */
        .stInfo {{
            background: var(--info-color) !important;
            color: white !important;
            border-color: var(--info-color) !important;
        }}

        /* Dataframes with dark theme */
        .stDataFrame, .stDataFrame table {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .stDataFrame th {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }}

        .stDataFrame td {{
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }}

        /* Code blocks */
        .stCode, pre {{
            background: var(--tertiary-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px;
        }}

        /* Plotly charts with theme support */
        .js-plotly-plot {{
            background: var(--card-bg) !important;
            border-radius: 8px;
        }}

        /* Enhanced Selectbox dropdown styling */
        .stSelectbox [data-baseweb="select"] {{
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        /* Selectbox input field */
        .stSelectbox [data-baseweb="select"] input {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        /* Selectbox dropdown container */
        .stSelectbox [data-baseweb="popover"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px var(--shadow) !important;
        }}

        /* Selectbox dropdown menu */
        .stSelectbox [role="listbox"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}

        /* Selectbox dropdown options */
        .stSelectbox [role="option"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            padding: 8px 12px !important;
        }}

        .stSelectbox [role="option"]:hover {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
        }}

        .stSelectbox [role="option"][aria-selected="true"] {{
            background: var(--accent-color) !important;
            color: white !important;
        }}

        /* Sidebar specific selectbox styling */
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
        .css-1d391kg .stSelectbox [data-baseweb="select"],
        .css-1cypcdb .stSelectbox [data-baseweb="select"],
        .css-1544g2n .stSelectbox [data-baseweb="select"] {{
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        section[data-testid="stSidebar"] .stSelectbox [role="listbox"],
        .css-1d391kg .stSelectbox [role="listbox"],
        .css-1cypcdb .stSelectbox [role="listbox"],
        .css-1544g2n .stSelectbox [role="listbox"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }}

        section[data-testid="stSidebar"] .stSelectbox [role="option"],
        .css-1d391kg .stSelectbox [role="option"],
        .css-1cypcdb .stSelectbox [role="option"],
        .css-1544g2n .stSelectbox [role="option"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        section[data-testid="stSidebar"] .stSelectbox [role="option"]:hover,
        .css-1d391kg .stSelectbox [role="option"]:hover,
        .css-1cypcdb .stSelectbox [role="option"]:hover,
        .css-1544g2n .stSelectbox [role="option"]:hover {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
        }}

        /* Enhanced File uploader styling */
        .stFileUploader {{
            background: var(--card-bg) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 8px;
            color: var(--text-primary) !important;
        }}

        .stFileUploader > div {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        .stFileUploader label {{
            color: var(--text-primary) !important;
        }}

        .stFileUploader [data-testid="stFileUploaderDropzone"] {{
            background: var(--secondary-bg) !important;
            border: 2px dashed var(--accent-color) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            padding: 2rem !important;
        }}

        .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] > div {{
            color: var(--text-secondary) !important;
        }}

        /* File uploader button */
        .stFileUploader button {{
            background: var(--accent-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }}

        .stFileUploader button:hover {{
            background: var(--accent-color) !important;
            opacity: 0.9 !important;
        }}

        /* Drag and drop area text */
        .stFileUploader div[data-testid="stFileUploaderDropzone"] > div {{
            color: var(--text-primary) !important;
        }}

        .stFileUploader div[data-testid="stFileUploaderDropzone"] small {{
            color: var(--text-secondary) !important;
        }}

        /* Slider */
        .stSlider > div > div > div > div {{
            background: var(--accent-color) !important;
        }}

        /* Custom components from our app */
        .ultra-header {{
            background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .metric-card {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        .metric-value {{
            color: var(--text-primary) !important;
        }}

        .metric-label {{
            color: var(--text-secondary) !important;
        }}

        .control-panel {{
            background: linear-gradient(145deg, var(--card-bg) 0%, var(--secondary-bg) 100%) !important;
            border: 2px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        .history-header {{
            background: linear-gradient(135deg, var(--secondary-bg) 0%, var(--tertiary-bg) 100%) !important;
            border: 3px solid var(--accent-color) !important;
            color: var(--text-primary) !important;
        }}

        .session-card {{
            background: linear-gradient(135deg, var(--card-bg) 0%, var(--secondary-bg) 100%) !important;
            border: 2px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        /* Additional sidebar components */
        .css-1d391kg .stRadio label, .css-1cypcdb .stRadio label, .css-1544g2n .stRadio label {{
            color: var(--text-primary) !important;
        }}

        .css-1d391kg .stTextInput label, .css-1cypcdb .stTextInput label, .css-1544g2n .stTextInput label {{
            color: var(--text-primary) !important;
        }}

        .css-1d391kg .stTextArea label, .css-1cypcdb .stTextArea label, .css-1544g2n .stTextArea label {{
            color: var(--text-primary) !important;
        }}

        /* Sidebar dividers */
        .css-1d391kg hr, .css-1cypcdb hr, .css-1544g2n hr {{
            border-color: var(--border-color) !important;
        }}

        /* Sidebar captions and help text */
        .css-1d391kg .stCaption, .css-1cypcdb .stCaption, .css-1544g2n .stCaption {{
            color: var(--text-secondary) !important;
        }}

        /* Sidebar metric containers */
        .css-1d391kg [data-testid="metric-container"],
        .css-1cypcdb [data-testid="metric-container"],
        .css-1544g2n [data-testid="metric-container"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        /* Ensure all text is visible with higher specificity */
        .stApp *, .css-1d391kg *, .css-1cypcdb *, .css-1544g2n * {{
            color: var(--text-primary) !important;
        }}

        /* Override for specific elements that should keep their colors */
        .stButton > button, .stSuccess, .stError, .stInfo,
        .stButton > button *, .stSuccess *, .stError *, .stInfo * {{
            color: white !important;
        }}

        .stWarning, .stWarning * {{
            color: var(--text-primary) !important;
        }}

        /* File uploader specific text */
        .stFileUploader *, .stFileUploader div, .stFileUploader span, .stFileUploader p {{
            color: var(--text-primary) !important;
        }}

        /* Enhanced selectbox dropdown options with higher specificity */
        .stSelectbox [role="listbox"],
        div[data-baseweb="popover"] [role="listbox"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px var(--shadow) !important;
        }}

        .stSelectbox [role="option"],
        div[data-baseweb="popover"] [role="option"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            padding: 8px 12px !important;
        }}

        .stSelectbox [role="option"]:hover,
        div[data-baseweb="popover"] [role="option"]:hover {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
        }}

        .stSelectbox [role="option"][aria-selected="true"],
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
            background: var(--accent-color) !important;
            color: white !important;
        }}

        /* Additional text visibility fixes */
        .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {{
            color: var(--text-primary) !important;
        }}

        /* Sidebar specific text fixes */
        .css-1d391kg p, .css-1d391kg span, .css-1d391kg div, .css-1d391kg label,
        .css-1cypcdb p, .css-1cypcdb span, .css-1cypcdb div, .css-1cypcdb label,
        .css-1544g2n p, .css-1544g2n span, .css-1544g2n div, .css-1544g2n label {{
            color: var(--text-primary) !important;
        }}

        /* File uploader text in main content */
        .stFileUploader p, .stFileUploader span, .stFileUploader div, .stFileUploader small {{
            color: var(--text-primary) !important;
        }}

        /* Drag and drop zone text */
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] small {{
            color: var(--text-primary) !important;
        }}

        /* Upload instructions text */
        [data-testid="stFileUploaderDropzoneInstructions"] p,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] div {{
            color: var(--text-primary) !important;
        }}

        /* Ensure all nested elements inherit proper colors */
        .stApp * {{
            color: inherit !important;
        }}

        /* Force text color for all elements */
        body, html, .stApp {{
            color: var(--text-primary) !important;
        }}

        /* Additional modern Streamlit component fixes */
        [data-testid="stSelectbox"] > div > div {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        /* Modern dropdown menu styling */
        [data-baseweb="menu"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px var(--shadow) !important;
        }}

        [data-baseweb="menu"] [role="option"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        [data-baseweb="menu"] [role="option"]:hover {{
            background: var(--secondary-bg) !important;
            color: var(--text-primary) !important;
        }}

        /* Sidebar markdown headers with better contrast */
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown h4,
        section[data-testid="stSidebar"] .stMarkdown h5,
        section[data-testid="stSidebar"] .stMarkdown h6 {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        /* Ensure all sidebar text is visible */
        section[data-testid="stSidebar"] * {{
            color: var(--text-primary) !important;
        }}

        /* Override for elements that should keep their specific colors */
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stButton > button * {{
            color: white !important;
        }}
        </style>
        """
        
        st.markdown(theme_css, unsafe_allow_html=True)

# Global theme manager instance
theme_manager = ThemeManager()
