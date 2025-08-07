import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from globals import xm_per_pix, time_window

# Initialize session state to store camera calibration and line objects
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.mtx, st.session_state.dist = None, None
    st.session_state.line_lt = Line(buffer_len=time_window)
    st.session_state.line_rt = Line(buffer_len=time_window)
    st.session_state.processed_frames = 0

def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Prepare the final output frame with visualizations and metrics
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road

def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset of vehicle from the center of the lane
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter

def process_frame(frame, keep_state=True):
    """
    Process a single frame through the lane detection pipeline
    """
    line_lt = st.session_state.line_lt
    line_rt = st.session_state.line_rt
    
    # Undistort the image
    img_undistorted = undistort(frame, st.session_state.mtx, st.session_state.dist, verbose=False)
    
    # Binarize the image
    img_binary = binarize(img_undistorted, verbose=False)
    
    # Apply bird's eye perspective transform
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)
    
    # Detect lane lines
    if st.session_state.processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)
    
    # Update line objects in session state
    st.session_state.line_lt = line_lt
    st.session_state.line_rt = line_rt
    
    # Compute offset from center
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])
    
    # Draw lane back onto the road
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    
    # Prepare output visualization
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)
    
    st.session_state.processed_frames += 1
    
    return blend_output

def process_video(input_path, keep_state=True):
    """
    Process a video file through the lane detection pipeline
    """
    # Get video properties
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video file
    output_path = f"{os.path.splitext(input_path)[0]}_processed.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Process each frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with lane detection
        try:
            processed_frame = process_frame(frame, keep_state=keep_state)
            out.write(processed_frame)
        except Exception as e:
            st.error(f"Error processing frame {i}: {str(e)}")
            continue
        
        # Update progress
        progress_percentage = (i + 1) / frame_count
        progress_bar.progress(progress_percentage)
        status_text.text(f"Processing: {int(progress_percentage * 100)}% complete")
    
    # Release resources
    cap.release()
    out.release()
    status_text.text("Processing complete!")
    
    return output_path

def reset_lane_detection():
    """Reset lane detection state"""
    st.session_state.line_lt = Line(buffer_len=time_window)
    st.session_state.line_rt = Line(buffer_len=time_window)
    st.session_state.processed_frames = 0

# Main Streamlit app
def main():
    st.set_page_config(page_title="Lane Morph", layout="wide")
    st.title("Lane Morph - Advanced Lane Detection")
    
    # Initialize camera calibration if not already done
    if st.session_state.mtx is None or st.session_state.dist is None:
        with st.spinner('Calibrating camera... This will take a moment.'):
            try:
                ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
                st.session_state.mtx = mtx
                st.session_state.dist = dist
                st.success("Camera calibration complete!")
            except Exception as e:
                st.error(f"Error during camera calibration: {str(e)}")
                st.stop()
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["Video Processing", "Image Processing", "About"])
    
    with tab1:
        st.header("Lane Detection in Videos")
        st.markdown("""
        Upload a road video to detect and highlight lane lines. The processed video will show 
        detected lanes, curvature measurements, and vehicle position.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            use_example = st.checkbox("Use example video")
        with col2:
            reset_button = st.button("Reset Lane Tracking")
            if reset_button:
                reset_lane_detection()
                st.success("Lane tracking reset!")
        
        if use_example:
            video_path = "project_video.mp4"
            if os.path.exists(video_path):
                st.video(video_path)
                
                if st.button("Process Example Video"):
                    with st.spinner("Processing video... This may take a while."):
                        processed_video_path = process_video(video_path, keep_state=True)
                        st.success("Processing complete!")
                        st.video(processed_video_path)
                        
                        with open(processed_video_path, "rb") as file:
                            st.download_button(
                                label="Download processed video",
                                data=file,
                                file_name="lane_detection_result.mp4",
                                mime="video/mp4"
                            )
            else:
                st.error(f"Example video not found at {video_path}")
        
        else:
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
            
            if uploaded_file is not None:
                # Save uploaded file to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile_path = tfile.name
                tfile.close()
                
                st.video(tfile_path)
                
                if st.button("Process Uploaded Video"):
                    with st.spinner("Processing video... This may take a while."):
                        processed_video_path = process_video(tfile_path, keep_state=True)
                        st.success("Processing complete!")
                        st.video(processed_video_path)
                        
                        with open(processed_video_path, "rb") as file:
                            st.download_button(
                                label="Download processed video",
                                data=file,
                                file_name="lane_detection_result.mp4",
                                mime="video/mp4"
                            )
    
    with tab2:
        st.header("Lane Detection in Images")
        st.markdown("""
        Upload a road image to detect and highlight lane lines.
        """)
        
        image_uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        with col1:
            use_example_img = st.checkbox("Use example image")
        with col2:
            if st.button("Reset Lane Tracking (Image)"):
                reset_lane_detection()
                st.success("Lane tracking reset!")
        
        if use_example_img:
            # Use an image from your test_images folder
            test_images_dir = "test_images"
            if os.path.exists(test_images_dir):
                test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if test_images:
                    selected_image = st.selectbox("Select example image", test_images)
                    image_path = os.path.join(test_images_dir, selected_image)
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    st.image(img_rgb, caption="Example Image")
                    
                    if st.button("Process Example Image"):
                        try:
                            with st.spinner("Processing image..."):
                                processed_img = process_frame(img, keep_state=False)
                                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                st.image(processed_img_rgb, caption="Processed Image")
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                else:
                    st.error(f"No example images found in {test_images_dir}")
            else:
                st.error(f"Example images directory not found at {test_images_dir}")
        
        elif image_uploaded is not None:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(image_uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption="Uploaded Image")
            
            if st.button("Process Uploaded Image"):
                try:
                    with st.spinner("Processing image..."):
                        processed_img = process_frame(img, keep_state=False)
                        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        st.image(processed_img_rgb, caption="Processed Image with Lane Detection")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    with tab3:
        st.header("About Lane Morph")
        st.markdown("""
        ## Advanced Lane Detection System
        
        Lane Morph uses computer vision techniques to detect and track lane lines in road images and videos.
        
        ### Algorithm Steps:
        1. **Camera Calibration**: Removes distortion effects from camera
        2. **Image Binarization**: Applies color and gradient thresholds to isolate lane markings
        3. **Bird's Eye View**: Applies perspective transform to get top-down view
        4. **Lane Detection**: Uses sliding window approach or previous frame data to detect lanes
        5. **Curvature Calculation**: Determines radius of curvature and vehicle position
        6. **Visualization**: Projects detected lanes back onto the road image
        
        ### Technologies Used:
        - OpenCV for image processing
        - NumPy for numerical operations
        - Streamlit for the web interface
        
        ### Created by:
        Ayush Bhatt ([@BhattAyush17](https://github.com/BhattAyush17/Lane_Morph))
        """)

if __name__ == "__main__":
    main()