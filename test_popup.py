import streamlit as st

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
                    st.session_state.export_popup_confirmed = True
                    st.rerun()
            
            with col2:
                if st.button("✕ Cancel", 
                           use_container_width=True,
                           key="popup_cancel_btn"):
                    st.session_state.show_export_popup = False
                    st.rerun()
        
        export_popup()

# Main app
st.title("🧪 Export Popup Test")

# Initialize session state
if 'show_export_popup' not in st.session_state:
    st.session_state.show_export_popup = False
if 'export_popup_confirmed' not in st.session_state:
    st.session_state.export_popup_confirmed = False

# Test button
if st.button("📊 Test Export Popup", type="primary"):
    show_export_popup_notification()

# Show confirmation message
if st.session_state.get('export_popup_confirmed', False):
    st.success("✅ Export confirmed! You clicked 'Go to Export Tab'")
    if st.button("Clear Message"):
        st.session_state.export_popup_confirmed = False
        st.rerun()
