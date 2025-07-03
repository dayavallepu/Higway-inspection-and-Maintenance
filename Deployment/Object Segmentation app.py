
                                     ################ Application for Object Detection and Segmentation ################



# Importing required libraries
import streamlit as st # streanlit for web app
from ultralytics import YOLO # YOLO for object detection and segmentation
import cv2 # OpenCV for image and video processing
import numpy as np # Numpy for numerical operations
import tempfile # Tempfile for temporary file handling

# Load the YOLOv8 model
model_path = r"D:\360 DigiTMG projects\Project-1\Highway inspection and Maintainance Using AI\8.Model Building\All_ClassMix_Annotation(mAP-97%)\runs\segment\train4\weights\best.pt"
model = YOLO(model_path) # Load the model

# Streamlit app title
st.title("Highway Inspection and Maintenance Using AI")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Run on Image", "Run on Video", "Live Video"]) # Select the app mode

# About the app
if app_mode == "About":
    st.markdown("""
    ## About This App
    This app uses **YOLOv8 for object detection and segmentation** in highway maintenance.  
    - Upload an **image** or **video** to see the model in action!  
    - Use the **Live Video** mode to see real-time segmentation from your webcam.
    """)

# Image detection and segmentation
elif app_mode == "Run on Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_container_width=True, channels="BGR")
        results = model(image)
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Processed Image", use_container_width=True, channels="BGR")

# Video detection and segmentation
elif app_mode == "Run on Video":
    st.header("Upload a Video")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Reset session variables when a new video is uploaded
        if "prev_video" not in st.session_state or st.session_state.prev_video != uploaded_file.name:
            st.session_state.frame_pos = 0  # Reset to start of video
            st.session_state.prev_video = uploaded_file.name  # Store new video name

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # Sidebar Controls
        frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 2) # Frame skip slider
        forward_skip = st.sidebar.slider("Forward Jump (frames)", 10, 500, 50) # Forward jump slider
        backward_skip = st.sidebar.slider("Backward Jump (frames)", 10, 500, 50) # Backward jump slider

        # Using session state for Pause, Forward, Backward, and Restart buttons
        if "paused" not in st.session_state:
            st.session_state.paused = False
        if "frame_pos" not in st.session_state:
            st.session_state.frame_pos = 0

        pause_button = st.sidebar.button("Pause/Resume") # Pause/Resume button
        forward_button = st.sidebar.button("Forward ⏩")    # Forward button
        backward_button = st.sidebar.button("Backward ⏪")  # Backward button
        restart_button = st.sidebar.button("Restart 🔄")    # Restart button

        if pause_button:
            st.session_state.paused = not st.session_state.paused  # Toggle pause

        if forward_button:
            st.session_state.frame_pos += forward_skip  # Jump forward

        if backward_button:
            st.session_state.frame_pos = max(0, st.session_state.frame_pos - backward_skip)  # Jump backward safely

        if restart_button:
            st.session_state.frame_pos = 0  # Reset video to start

        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)  # Set video to correct frame

        while cap.isOpened():
            if not st.session_state.paused:
                ret, frame = cap.read()
                if not ret:
                    break

                st.session_state.frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                if int(st.session_state.frame_pos) % frame_skip == 0:
                    frame_resized = cv2.resize(frame, (640, 640))
                    results = model(frame_resized)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()

# Live video detection and segmentation
elif app_mode == "Live Video": 
    st.header("Live Video Detection & Segmentation")

    run_live = st.checkbox("Start Live Detection")

    if run_live:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 2)
        
        if "paused" not in st.session_state:
            st.session_state.paused = False
        
        pause_button = st.sidebar.button("Pause/Resume")

        if pause_button:
            st.session_state.paused = not st.session_state.paused  

        frame_count = 0

        while run_live:
            if not st.session_state.paused:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break

                if frame_count % frame_skip == 0:
                    frame_resized = cv2.resize(frame, (640, 640))
                    results = model(frame_resized)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                frame_count += 1  

        cap.release()


