"""
PPE Detection Results Viewer
Comprehensive display and analysis of video processing results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
from datetime import datetime


def create_results_dashboard(results, output_path, processing_time, settings, unique_id=None):
    """Create a comprehensive results dashboard"""

    # Generate unique identifier for charts if not provided
    if unique_id is None:
        unique_id = str(hash(str(results) + str(processing_time)))[:8]
    
    st.markdown("---")

    # Enhanced section header for detection results with better visual design
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.10) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    ">
        <h2 style="
            margin: 0;
            color: #2c3e50;
            font-size: 2.2rem;
            font-weight: 800;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            letter-spacing: 1px;
        ">🎯 DETECTION RESULTS</h2>
        <p style="
            margin: 1rem 0 0 0;
            color: #64748b;
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 500;
        ">Comprehensive analysis results displayed in broader horizontal tabs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing summary header
    speed_boost = settings.get('skip', 1)
    total_frames = results['processed_frames']
    fps = total_frames / processing_time if processing_time > 0 else 0
    
    # Success message with details
    st.success(f"""
    ✅ **Video Analysis Complete!**
    - Processing Time: {processing_time:.1f} seconds
    - Speed Boost: {speed_boost}x faster
    - Processing Rate: {fps:.1f} FPS
    """)
    
    # Key metrics in prominent cards with improved styling
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    # Add responsive CSS for better layout
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, var(--card-color) 0%, var(--card-color-dark) 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 140px;
        max-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        word-wrap: break-word;
        overflow-wrap: break-word;
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.15);
        cursor: pointer;
        backdrop-filter: blur(15px);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.6s ease;
        opacity: 0;
    }

    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.3);
    }

    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-card:hover::after {
        opacity: 1;
        transform: rotate(45deg) translate(50%, 50%);
    }
    .metric-value {
        font-size: clamp(1.8rem, 3.5vw, 2.8rem);
        font-weight: 900;
        margin: 0 0 0.8rem 0;
        line-height: 0.95;
        text-shadow: 0 3px 6px rgba(0,0,0,0.4);
        word-break: keep-all;
        text-align: center;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
        position: relative;
        z-index: 2;
        letter-spacing: -0.5px;
        display: block;
    }

    .metric-label {
        font-size: clamp(0.75rem, 1.8vw, 1.1rem);
        font-weight: 700;
        margin: 0;
        opacity: 0.98;
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.2;
        word-break: break-word;
        hyphens: auto;
        text-align: center;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        max-width: 100%;
        position: relative;
        z-index: 2;
        padding: 0 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    @media (max-width: 1200px) {
        .metric-card {
            padding: 1.8rem 1.2rem;
            min-height: 120px;
            max-height: 140px;
        }
        .metric-value {
            font-size: clamp(1.6rem, 3.2vw, 2.4rem);
            margin: 0 0 0.6rem 0;
        }
        .metric-label {
            font-size: clamp(0.7rem, 1.6vw, 1rem);
        }
    }

    @media (max-width: 768px) {
        .metric-card {
            padding: 1.5rem 1rem;
            min-height: 100px;
            max-height: 120px;
            margin: 0.6rem 0;
        }
        .metric-value {
            font-size: clamp(1.4rem, 3vw, 2rem);
            margin: 0 0 0.5rem 0;
        }
        .metric-label {
            font-size: clamp(0.65rem, 1.4vw, 0.9rem);
            letter-spacing: 0.8px;
            padding: 0 0.3rem;
        }
    }

    @media (max-width: 480px) {
        .metric-card {
            padding: 1.2rem 0.8rem;
            min-height: 85px;
            max-height: 100px;
            margin: 0.4rem 0;
        }
        .metric-value {
            font-size: clamp(1.2rem, 2.8vw, 1.6rem);
            margin: 0 0 0.4rem 0;
        }
        .metric-label {
            font-size: clamp(0.6rem, 1.2vw, 0.8rem);
            letter-spacing: 0.6px;
            padding: 0 0.2rem;
            -webkit-line-clamp: 1;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="--card-color: #4CAF50; --card-color-dark: #45a049;">
            <div class="metric-value">{results['processed_frames']:,}</div>
            <div class="metric-label">Total Frames</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        violations_color = "#f44336" if results['total_violations'] > 0 else "#4CAF50"
        violations_color_dark = "#d32f2f" if results['total_violations'] > 0 else "#45a049"
        st.markdown(f"""
        <div class="metric-card" style="--card-color: {violations_color}; --card-color-dark: {violations_color_dark};">
            <div class="metric-value">{results['total_violations']:,}</div>
            <div class="metric-label">Violations Found</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_compliance = results.get('average_compliance_rate', 0)
        compliance_color = "#4CAF50" if avg_compliance >= 80 else "#ff9800" if avg_compliance >= 60 else "#f44336"
        compliance_color_dark = "#45a049" if avg_compliance >= 80 else "#f57c00" if avg_compliance >= 60 else "#d32f2f"
        st.markdown(f"""
        <div class="metric-card" style="--card-color: {compliance_color}; --card-color-dark: {compliance_color_dark};">
            <div class="metric-value">{avg_compliance:.0f}%</div>
            <div class="metric-label">Avg Compliance</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        detection_frames = results.get('detection_frames', 0)
        st.markdown(f"""
        <div class="metric-card" style="--card-color: #2196F3; --card-color-dark: #1976D2;">
            <div class="metric-value">{detection_frames:,}</div>
            <div class="metric-label">Frames Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced broader tab styling for perfect visual design
    st.markdown("""
    <style>
    /* Enhanced broader tab styling for detection results */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        border-radius: 24px;
        padding: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .stTabs [data-baseweb="tab"] {
        height: 65px;
        min-width: 220px;
        padding: 0 2.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-radius: 18px;
        color: var(--text-color, #333);
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        position: relative;
        letter-spacing: 0.5px;
    }

    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 18px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border-color: rgba(102, 126, 234, 0.6);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }

    .stTabs [data-baseweb="tab"]:hover::before {
        opacity: 1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
        color: white;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
        transform: translateY(-4px);
        font-weight: 800;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* Enhanced responsive design for broader tabs */
    @media (max-width: 1200px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 180px;
            padding: 0 2rem;
            font-size: 1rem;
            height: 60px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            padding: 16px;
        }
    }

    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            padding: 14px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            padding: 0 1.5rem;
            min-width: 150px;
            height: 55px;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 2rem;
        }
    }

    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0 1rem;
            min-width: 130px;
            height: 50px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 12px;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Clean results tabs with shorter names (removed Video tab per user request)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analysis",
        "⚠️ Violations",
        "📈 Stats",
        "📥 Export"
    ])

    with tab1:
        display_compliance_analysis(results, settings, unique_id)

    with tab2:
        display_violation_details(results, unique_id)

    with tab3:
        display_statistics(results, processing_time, settings)

    with tab4:
        display_download_options(results, output_path, processing_time, settings, unique_id)


def display_processed_video(output_path, unique_id=None):
    """Display the processed video with controls"""

    st.markdown("### 📹 Processed Video with PPE Detection")

    video_data = None
    file_size = 0

    # Try to get video from file first (with better error handling)
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as video_file:
                video_data = video_file.read()
            file_size = os.path.getsize(output_path) / (1024 * 1024)
        except Exception as e:
            st.warning(f"⚠️ Could not read video file: {str(e)}")
            video_data = None

    # Fallback to session state if file doesn't exist or couldn't be read
    if not video_data and hasattr(st.session_state, 'processed_video_data') and st.session_state.processed_video_data:
        try:
            video_data = st.session_state.processed_video_data
            file_size = len(video_data) / (1024 * 1024)
            st.info("📹 Displaying video from session memory")
        except Exception as e:
            st.warning(f"⚠️ Could not load video from session: {str(e)}")
            video_data = None

    if video_data:
        # Video player
        st.video(video_data)

        # Enhanced video information and controls
        st.markdown("#### 📊 Video Information & Controls")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 **File Size:** {file_size:.1f} MB")
        with col2:
            st.info(f"📍 **Location:** {os.path.basename(output_path) if output_path else 'Session Memory'}")

        # Download section
        st.markdown("#### 📥 Download Processed Video")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            download_key = f"download_video_processed_{unique_id}" if unique_id else "download_video_processed"
            st.download_button(
                "📹 Download MP4 Video",
                video_data,
                file_name=f"ppe_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                use_container_width=True,
                help="Download the processed video with PPE detection overlays",
                key=download_key
            )

        with col_dl2:
            # Video quality info
            st.markdown(f"""
            **📊 Video Details:**
            - **Format**: MP4 (H.264)
            - **Quality**: Original resolution
            - **Features**: PPE detection overlays
            - **Size**: {file_size:.1f} MB
            """)

        # Video controls info
        st.markdown("""
        **🎮 Video Controls:**
        - ▶️ Click play/pause to control playback
        - 🎯 Use the timeline to jump to specific moments
        - 🔊 Adjust volume as needed
        - 📱 Works on mobile devices
        """)

    else:
        st.error("❌ Processed video not available")
        st.markdown("""
        **Possible reasons:**
        - Video file was moved or deleted
        - Processing was interrupted
        - Insufficient storage space

        **💡 Tip:** Try processing the video again or check the Results History tab.
        """)


def display_compliance_analysis(results, settings, unique_id=None):
    """Display detailed compliance analysis"""
    
    st.markdown("### 📊 Compliance Rate Analysis")
    
    if results.get('compliance_timeline'):
        timeline = results['compliance_timeline']
        frames = list(range(0, len(timeline) * settings.get('skip', 1), settings.get('skip', 1)))
        
        # Create compliance chart
        fig = go.Figure()
        
        # Main compliance line
        fig.add_trace(go.Scatter(
            x=frames,
            y=timeline,
            mode='lines+markers',
            name='Compliance Rate',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6),
            hovertemplate='Frame: %{x}<br>Compliance: %{y:.1f}%<extra></extra>'
        ))
        
        # Threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="green", 
                     annotation_text="Good Compliance (80%)")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Fair Compliance (60%)")
        fig.add_hline(y=40, line_dash="dash", line_color="red", 
                     annotation_text="Poor Compliance (40%)")
        
        fig.update_layout(
            title="Compliance Rate Throughout Video",
            xaxis_title="Frame Number",
            yaxis_title="Compliance Rate (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        chart_id = f"compliance_timeline_chart_{unique_id}" if unique_id else "compliance_timeline_chart"
        st.plotly_chart(fig, use_container_width=True, key=chart_id)
        
        # Compliance statistics with improved styling
        st.markdown("#### 📈 Compliance Statistics")

        col1, col2, col3, col4 = st.columns(4)

        # Add responsive CSS for compliance stats
        st.markdown("""
        <style>
        .compliance-stat-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
            padding: 1.2rem 1rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            border-left: 4px solid var(--border-color);
            border-right: 1px solid rgba(0,0,0,0.05);
            border-top: 1px solid rgba(0,0,0,0.05);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            margin: 0.5rem 0;
            word-wrap: break-word;
            overflow-wrap: break-word;
            transition: all 0.3s ease;
            cursor: pointer;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .compliance-stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border-left-width: 5px;
        }

        .compliance-stat-value {
            font-size: clamp(1.3rem, 2.5vw, 1.8rem);
            font-weight: 900;
            margin: 0 0 0.5rem 0;
            color: var(--text-primary);
            line-height: 1;
            word-break: keep-all;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }

        .compliance-stat-label {
            font-size: clamp(0.65rem, 1.2vw, 0.85rem);
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin: 0;
            line-height: 1.3;
            word-break: break-word;
            hyphens: auto;
            text-align: center;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            max-width: 100%;
            padding: 0 0.2rem;
        }
        @media (max-width: 768px) {
            .compliance-stat-card {
                padding: 0.6rem;
            }
            .compliance-stat-value {
                font-size: 1.1rem;
            }
            .compliance-stat-label {
                font-size: 0.65rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        with col1:
            peak_compliance = max(timeline)
            st.markdown(f"""
            <div class="compliance-stat-card" style="--border-color: #4CAF50;">
                <div class="compliance-stat-value">{peak_compliance:.1f}%</div>
                <div class="compliance-stat-label">Peak Compliance</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            lowest_compliance = min(timeline)
            st.markdown(f"""
            <div class="compliance-stat-card" style="--border-color: #ff9800;">
                <div class="compliance-stat-value">{lowest_compliance:.1f}%</div>
                <div class="compliance-stat-label">Lowest Compliance</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            std_dev = np.std(timeline)
            st.markdown(f"""
            <div class="compliance-stat-card" style="--border-color: #2196F3;">
                <div class="compliance-stat-value">{std_dev:.1f}%</div>
                <div class="compliance-stat-label">Standard Deviation</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            good_frames = sum(1 for rate in timeline if rate >= 80)
            good_percentage = (good_frames / len(timeline)) * 100
            st.markdown(f"""
            <div class="compliance-stat-card" style="--border-color: #9c27b0;">
                <div class="compliance-stat-value">{good_percentage:.1f}%</div>
                <div class="compliance-stat-label">Good Compliance</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Compliance distribution
        st.markdown("#### 📊 Compliance Distribution")
        
        # Create histogram
        fig_hist = go.Figure(data=[go.Histogram(
            x=timeline,
            nbinsx=20,
            marker_color='#667eea',
            opacity=0.7
        )])
        
        fig_hist.update_layout(
            title="Distribution of Compliance Rates",
            xaxis_title="Compliance Rate (%)",
            yaxis_title="Number of Frames",
            template="plotly_white",
            height=400
        )
        
        chart_id = f"compliance_distribution_chart_{unique_id}" if unique_id else "compliance_distribution_chart"
        st.plotly_chart(fig_hist, use_container_width=True, key=chart_id)
        
    else:
        st.info("📊 No compliance timeline data available for this video.")


def display_violation_details(results, unique_id=None):
    """Display detailed violation information"""
    
    st.markdown("### ⚠️ Violation Analysis")
    
    if results.get('frame_violations'):
        violations = results['frame_violations']
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
            <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700;">
                🚨 Total violation instances found: {len(violations)}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Violation summary table
        st.markdown("#### 📋 Violation Timeline")
        
        violation_data = []
        for i, v in enumerate(violations):
            frame_num = v['frame']
            timestamp = v['timestamp']
            violation_types = [viol['type'] for viol in v['violations']]
            violation_count = len(v['violations'])
            confidence_scores = [f"{viol['confidence']:.2f}" for viol in v['violations']]
            
            violation_data.append({
                'Index': i + 1,
                'Frame': frame_num,
                'Time (seconds)': f"{timestamp:.1f}",
                'Violations Count': violation_count,
                'Violation Types': ', '.join(violation_types),
                'Confidence Scores': ', '.join(confidence_scores)
            })
        
        df = pd.DataFrame(violation_data)
        
        # Display with pagination
        if len(df) > 20:
            st.markdown(f"**Showing first 20 violations out of {len(df)} total**")
            st.dataframe(df.head(20), use_container_width=True)
            
            if st.button("📄 Show All Violations"):
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        
        # Violation type analysis
        st.markdown("#### 📈 Violation Type Analysis")
        
        violation_counts = {}
        total_violations = 0
        
        for v in violations:
            for viol in v['violations']:
                v_type = viol['type']
                violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
                total_violations += 1
        
        if violation_counts:
            # Pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=list(violation_counts.values()),
                    names=list(violation_counts.keys()),
                    title="Distribution of Violation Types",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=400)
                chart_id = f"violation_pie_chart_{unique_id}" if unique_id else "violation_pie_chart"
                st.plotly_chart(fig_pie, use_container_width=True, key=chart_id)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    x=list(violation_counts.keys()),
                    y=list(violation_counts.values()),
                    title="Violation Count by Type",
                    color=list(violation_counts.values()),
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(height=400)
                chart_id = f"violation_bar_chart_{unique_id}" if unique_id else "violation_bar_chart"
                st.plotly_chart(fig_bar, use_container_width=True, key=chart_id)
            
            # Detailed breakdown with improved styling
            st.markdown("#### 📊 Detailed Breakdown")

            for v_type, count in violation_counts.items():
                percentage = (count / total_violations) * 100
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0;
                            border-left: 3px solid #dc3545;">
                    <strong style="color: #2c3e50; font-size: 1rem;">{v_type}</strong>:
                    <span style="color: #495057;">{count} instances ({percentage:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                    color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 1rem 0;
                    box-shadow: 0 6px 25px rgba(76, 175, 80, 0.3);">
            <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 700;">
                ✅ Excellent! No violations detected in this video.
            </h2>
            <div style="font-size: 1rem; line-height: 1.6; opacity: 0.95;">
                <strong>This means:</strong><br>
                • All detected people were wearing required PPE<br>
                • Workplace safety compliance is maintained<br>
                • No immediate safety concerns identified
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_statistics(results, processing_time, settings):
    """Display comprehensive statistics"""
    
    st.markdown("### 📈 Processing Statistics")
    
    # Performance metrics with improved styling
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚡ Performance Metrics")

        total_frames = results['processed_frames']
        detection_frames = results.get('detection_frames', 0)
        skip_frames = settings.get('skip', 1)

        # Create styled performance metrics
        metrics = [
            ("Total Frames", f"{total_frames:,}", "#667eea"),
            ("Frames Analyzed", f"{detection_frames:,}", "#4CAF50"),
            ("Frames Skipped", f"{total_frames - detection_frames:,}", "#ff9800"),
            ("Processing Time", f"{processing_time:.1f} seconds", "#2196F3"),
            ("Processing Speed", f"{total_frames / processing_time:.1f} FPS" if processing_time > 0 else "N/A", "#9c27b0"),
            ("Speed Boost", f"{skip_frames}x", "#f44336"),
            ("Average FPS", f"{detection_frames / processing_time:.1f} FPS" if processing_time > 0 else "N/A", "#00bcd4")
        ]

        # Add responsive CSS for metric rows
        st.markdown("""
        <style>
        .metric-row {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.8) 100%);
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin: 0.4rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-left: 4px solid var(--metric-color);
            border-right: 1px solid rgba(0,0,0,0.05);
            border-top: 1px solid rgba(0,0,0,0.05);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
            word-wrap: break-word;
            overflow-wrap: break-word;
            transition: all 0.3s ease;
            cursor: pointer;
            min-height: 60px;
        }

        .metric-row:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            border-left-width: 5px;
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
        }
        .metric-name {
            font-weight: 700;
            color: var(--text-primary);
            font-size: clamp(0.85rem, 1.8vw, 1rem);
            flex: 1;
            margin-right: 1rem;
            word-break: break-word;
            hyphens: auto;
            text-shadow: 0 1px 2px rgba(0,0,0,0.05);
            line-height: 1.4;
        }

        .metric-value {
            font-weight: 900;
            color: var(--metric-color);
            font-size: clamp(0.9rem, 2vw, 1.1rem);
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 45%;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            letter-spacing: 0.3px;
            line-height: 1.2;
        }
        @media (max-width: 768px) {
            .metric-row {
                padding: 0.5rem;
                flex-direction: column;
                align-items: flex-start;
                text-align: left;
            }
            .metric-name {
                margin-right: 0;
                margin-bottom: 0.2rem;
                font-size: 0.8rem;
            }
            .metric-value {
                font-size: 0.9rem;
                text-align: left;
                max-width: 100%;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        for metric_name, metric_value, color in metrics:
            st.markdown(f"""
            <div class="metric-row" style="--metric-color: {color};">
                <span class="metric-name">{metric_name}</span>
                <span class="metric-value">{metric_value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Detection Statistics")

        avg_compliance = results.get('average_compliance_rate', 0)
        total_violations = results['total_violations']

        # Create styled detection metrics
        grade = "A" if avg_compliance >= 90 else "B" if avg_compliance >= 80 else "C" if avg_compliance >= 70 else "D"
        status = "Excellent" if avg_compliance >= 90 else "Good" if avg_compliance >= 80 else "Fair" if avg_compliance >= 70 else "Poor"
        grade_color = "#4CAF50" if grade in ["A", "B"] else "#ff9800" if grade == "C" else "#f44336"

        detection_metrics = [
            ("Average Compliance Rate", f"{avg_compliance:.1f}%", "#4CAF50"),
            ("Total Violations", f"{total_violations:,}", "#f44336"),
            ("Violation Rate", f"{(total_violations / detection_frames * 100):.2f}%" if detection_frames > 0 else "0%", "#ff9800"),
            ("Compliance Grade", grade, grade_color),
            ("Safety Status", status, grade_color)
        ]

        for metric_name, metric_value, color in detection_metrics:
            st.markdown(f"""
            <div class="metric-row" style="--metric-color: {color};">
                <span class="metric-name">{metric_name}</span>
                <span class="metric-value">{metric_value}</span>
            </div>
            """, unsafe_allow_html=True)


def display_download_options(results, output_path, processing_time, settings, unique_id=None):
    """Display comprehensive download options"""

    st.markdown("### 📥 Download All Results")

    st.markdown("""
    **📦 Available Downloads:**
    Get all your processed results in multiple formats for documentation, analysis, and sharing.
    """)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📹 Video Downloads")

        # Download processed video
        video_data = None

        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                video_data = f.read()
        elif hasattr(st.session_state, 'processed_video_data') and st.session_state.processed_video_data:
            video_data = st.session_state.processed_video_data

        if video_data:
            file_size = len(video_data) / (1024 * 1024)

            download_key = f"download_video_{unique_id}" if unique_id else "download_video"
            st.download_button(
                "📹 **Download Processed Video**",
                video_data,
                file_name=f"ppe_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                use_container_width=True,
                help=f"Download the video with PPE detection overlays ({file_size:.1f} MB)",
                key=download_key
            )

            st.success(f"✅ Video ready ({file_size:.1f} MB)")
        else:
            st.error("❌ Video file not available")
            st.info("💡 Try processing a video first")
    
    with col2:
        st.markdown("#### 📊 Data Downloads")
        
        # Download detailed report
        report_data = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'speed_boost': settings.get('skip', 1),
                'settings': settings
            },
            'summary': {
                'total_frames': results['processed_frames'],
                'detection_frames': results.get('detection_frames', 0),
                'total_violations': results['total_violations'],
                'average_compliance_rate': results.get('average_compliance_rate', 0)
            },
            'violations': results.get('frame_violations', []),
            'compliance_timeline': results.get('compliance_timeline', [])
        }
        
        download_key = f"download_json_{unique_id}" if unique_id else "download_json"
        st.download_button(
            "📄 **Download Full Report (JSON)**",
            json.dumps(report_data, indent=2),
            file_name=f"ppe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Complete analysis data in JSON format",
            key=download_key
        )
    
    with col3:
        st.markdown("#### 📋 Summary Downloads")
        
        # Download violation summary CSV
        if results.get('frame_violations'):
            violation_summary = []
            for v in results['frame_violations']:
                violation_summary.append({
                    'Frame': v['frame'],
                    'Timestamp_Seconds': f"{v['timestamp']:.1f}",
                    'Violation_Count': len(v['violations']),
                    'Violation_Types': ', '.join([viol['type'] for viol in v['violations']]),
                    'Confidence_Scores': ', '.join([f"{viol['confidence']:.2f}" for viol in v['violations']])
                })
            
            df = pd.DataFrame(violation_summary)
            csv_data = df.to_csv(index=False)
            
            download_key = f"download_csv_{unique_id}" if unique_id else "download_csv"
            st.download_button(
                "📊 **Download Violations (CSV)**",
                csv_data,
                file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Violation data in spreadsheet format",
                key=download_key
            )
        else:
            st.success("✅ No violations to download")
    
    # Additional download info
    st.markdown("---")
    st.markdown("#### ℹ️ Download Information")
    
    st.info("""
    **📁 File Formats:**
    - **MP4 Video**: Processed video with detection overlays
    - **JSON Report**: Complete analysis data for further processing
    - **CSV Data**: Violation data for spreadsheet analysis
    
    **💡 Tips:**
    - JSON files can be imported into other analysis tools
    - CSV files open directly in Excel or Google Sheets
    - Videos can be shared for review or documentation
    """)
