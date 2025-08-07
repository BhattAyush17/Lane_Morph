# Lane Morph - Advanced Lane Detection System

Lane Detection [![Watch the video](https://img.youtube.com/vi/<[[VIDEO_ID](https://youtu.be/tYuC7CDGDyc?si=PUQOaQ19TMdm-i1x)](https://youtu.be/tYuC7CDGDyc?si=PUQOaQ19TMdm-i1x)>/hqdefault.jpg)](https://www.youtube.com/watch?v=<VIDEO_ID>)


An advanced lane detection system that uses computer vision techniques to identify lane boundaries in road videos and images. The system calculates lane curvature, vehicle position, and provides visual feedback.

## Features

- Camera calibration and distortion correction using chessboard images
- Color and gradient thresholding for robust lane line detection
- Perspective transformation to bird's eye view
- Lane detection using sliding window and polynomial fitting approaches
- Radius of curvature calculation and vehicle position estimation
- Real-time visualization with metrics overlay
- Support for both image and video processing

## Installation

```bash
# Clone the repository
git clone https://github.com/BhattAyush17/Lane_Morph.git
cd Lane_Morph

# Install required packages
pip install -r requirements.txt


Usage
Web Interface
The project includes a Streamlit web interface for easy interaction:

bash
streamlit run app.py
Navigate to the displayed URL in your browser to:

Upload and process your own road videos
Process individual images
Use example videos included in the project
View lane detection metrics in real-time
Command Line
You can also run the pipeline from the command line:

bash
# For video processing
python main.py

# For image processing (modify main.py mode variable to 'image')
python main.py
Project Structure
Code
Lane_Morph/
├── app.py                  # Streamlit web application
├── main.py                 # Main pipeline for lane detection
├── binarization_utils.py   # Utilities for image thresholding
├── calibration_utils.py    # Camera calibration functions
├── globals.py              # Global constants and parameters
├── line_utils.py           # Lane line detection algorithms
├── perspective_utils.py    # Perspective transformation utilities
├── camera_cal/             # Calibration images for camera calibration
├── test_images/            # Test images for lane detection
├── output_images/          # Processed output images
├── requirements.txt        # Required Python packages
└── README.md               # This documentation file
Technologies Used
Python 3.8+
OpenCV for computer vision operations
NumPy for numerical calculations
Matplotlib for visualization
Streamlit for web interface
Moviepy for video processing
Future Improvements
Improved robustness in challenging lighting conditions
Support for sharper turns and complex road scenarios
Real-time processing optimization
Lane departure warning system
Integration with other driving assistance systems
Acknowledgements
OpenCV documentation and tutorials
Udacity Self-Driving Car Nanodegree resources
Computer Vision research papers on lane detection
License
MIT License

Code

## .gitignore File

```text name=.gitignore
# Python related
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
*.manifest
*.spec
.ipynb_checkpoints

# Editor directories and files
.idea/
.vscode/
*.swp
*.swo
.vs/
.spyderproject
.spyproject
.ropeproject

# Project specific
_pycache_/
.DS_Store
*.pickle
calibration_pickle.p

# Optional - uncomment if you want to exclude large video files
# *.mp4

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Streamlit
.streamlit/

# Output directories that might contain large files
# Uncomment if needed
# output_videos/
