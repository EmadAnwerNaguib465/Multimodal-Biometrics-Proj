import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modules.authentication import authenticate_user
from modules.liveness_detection import check_liveness
from modules.face_recognition import save_face_embedding, extract_face_embedding
from modules.fingerprint_recognition import save_fingerprint_template
import os

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Biometric Authentication",
    page_icon="🔐",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'enrollment_mode' not in st.session_state:
    st.session_state.enrollment_mode = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user"

# Title and description
st.title("🔐 Multi-Modal Biometric Authentication System")
st.markdown("### Secure access using face recognition, fingerprint matching, and liveness detection")

# Sidebar for mode selection
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Select Mode",
        ["Authentication", "Enrollment"],
        index=0 if not st.session_state.enrollment_mode else 1
    )
    st.session_state.enrollment_mode = (mode == "Enrollment")
    
    st.divider()
    
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id
    
    st.divider()
    
    st.markdown("### 📊 System Info")
    st.info("""
    **Authentication Modes:**
    - Face + Fingerprint (biometric)
    - Password (fallback)
    
    **Security Features:**
    - Liveness detection
    - Multi-factor authentication
    - Encrypted templates
    """)

# Main content area
if st.session_state.enrollment_mode:
    # ===== ENROLLMENT MODE =====
    st.header("📝 Enrollment Mode")
    st.info("Register your biometric data for authentication")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Face Enrollment")
        face_enroll = st.camera_input("Capture your face")
        
        if face_enroll and st.button("💾 Save Face Template", key="save_face"):
            with st.spinner("Processing face..."):
                try:
                    # Convert to numpy array
                    image = Image.open(face_enroll)
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Extract and save embedding
                    embedding = extract_face_embedding(img_bgr)
                    if embedding is not None:
                        save_face_embedding(embedding, st.session_state.user_id)
                        st.success(f"✅ Face template saved for user: {st.session_state.user_id}")
                    else:
                        st.error("❌ Could not detect face in image")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("👆 Fingerprint Enrollment")
        fingerprint_enroll = st.file_uploader(
            "Upload fingerprint image",
            type=["png", "jpg", "jpeg"],
            key="fingerprint_enroll"
        )
        
        if fingerprint_enroll and st.button("💾 Save Fingerprint Template", key="save_fingerprint"):
            with st.spinner("Processing fingerprint..."):
                try:
                    # Convert to numpy array
                    image = Image.open(fingerprint_enroll)
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        img_gray = img_array
                    
                    # Save template
                    save_fingerprint_template(img_gray, st.session_state.user_id)
                    st.success(f"✅ Fingerprint template saved for user: {st.session_state.user_id}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    # ===== AUTHENTICATION MODE =====
    st.header("🔓 Authentication Mode")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📸 Biometric Login", "🔑 Password Login", "🔒 Multi-Factor"])
    
    with tab1:
        st.subheader("Biometric Authentication")
        
        col1, col2 = st.columns(2)
        
        with col1:
            face_img = st.camera_input("📸 Capture Face")
        
        with col2:
            fingerprint_img = st.file_uploader(
                "👆 Upload Fingerprint",
                type=["png", "jpg", "jpeg"]
            )
        
        if st.button("🚀 Authenticate with Biometrics", key="auth_bio"):
            if not face_img and not fingerprint_img:
                st.warning("⚠️ Please provide at least one biometric input")
            else:
                with st.spinner("Verifying..."):
                    # Liveness check for face
                    if face_img:
                        image = Image.open(face_img)
                        img_array = np.array(image)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        is_live = check_liveness(img_bgr)
                        if not is_live:
                            st.error("🚨 Liveness check failed! Face appears to be spoofed.")
                            st.stop()
                        else:
                            st.success("✅ Liveness check passed")
                    
                    # Authenticate
                    result = authenticate_user(
                        face_img,
                        fingerprint_img,
                        None,
                        user_id=st.session_state.user_id,
                        require_biometric=True
                    )
                    
                    if result['authenticated']:
                        st.success("✅ Authentication Successful!")
                        st.balloons()
                        
                        # Show details
                        with st.expander("📋 Authentication Details"):
                            st.json(result['details'])
                        
                        st.session_state.authenticated = True
                    else:
                        st.error("❌ Authentication Failed")
                        with st.expander("📋 Failure Details"):
                            st.json(result)
    
    with tab2:
        st.subheader("Password Fallback")
        password = st.text_input("🔑 Enter Password", type="password", key="pwd_only")
        
        if st.button("🚀 Login with Password", key="auth_pwd"):
            if not password:
                st.warning("⚠️ Please enter a password")
            else:
                with st.spinner("Verifying..."):
                    result = authenticate_user(
                        None,
                        None,
                        password,
                        user_id=st.session_state.user_id,
                        require_biometric=False
                    )
                    
                    if result['authenticated']:
                        st.success("✅ Authentication Successful!")
                        st.session_state.authenticated = True
                    else:
                        st.error("❌ Authentication Failed")
    
    with tab3:
        st.subheader("Multi-Factor Authentication")
        st.info("All factors must pass for authentication")
        
        col1, col2 = st.columns(2)
        
        with col1:
            face_mfa = st.camera_input("📸 Capture Face", key="face_mfa")
        
        with col2:
            fingerprint_mfa = st.file_uploader(
                "👆 Upload Fingerprint",
                type=["png", "jpg", "jpeg"],
                key="fp_mfa"
            )
        
        password_mfa = st.text_input("🔑 Enter Password", type="password", key="pwd_mfa")
        
        if st.button("🚀 Authenticate (All Factors)", key="auth_mfa"):
            if not (face_mfa or fingerprint_mfa or password_mfa):
                st.warning("⚠️ Please provide at least one authentication factor")
            else:
                with st.spinner("Verifying all factors..."):
                    # Liveness check
                    if face_mfa:
                        image = Image.open(face_mfa)
                        img_array = np.array(image)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        is_live = check_liveness(img_bgr)
                        if not is_live:
                            st.error("🚨 Liveness check failed!")
                            st.stop()
                    
                    # Authenticate with all factors required
                    result = authenticate_user(
                        face_mfa,
                        fingerprint_mfa,
                        password_mfa,
                        user_id=st.session_state.user_id,
                        require_all=True
                    )
                    
                    if result['authenticated']:
                        st.success("✅ Multi-Factor Authentication Successful!")
                        st.balloons()
                        
                        with st.expander("📋 Authentication Details"):
                            st.json(result)
                        
                        st.session_state.authenticated = True
                    else:
                        st.error("❌ Authentication Failed")
                        st.warning(f"Passed: {result['factors_passed_count']}/{result['factors_attempted']} factors")
                        with st.expander("📋 Failure Details"):
                            st.json(result)

# Show authenticated state
if st.session_state.authenticated:
    st.success("🎉 You are currently authenticated!")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    🔐 Multi-Modal Biometric Authentication System v1.0<br>
    Secure • Fast • Reliable
</div>
""", unsafe_allow_html=True)