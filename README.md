# Car Counting System Using YOLO

This repository contains a Python project for counting cars in a video stream using the YOLO (You Only Look Once) object detection model. The project is designed to detect and count cars passing through a designated line on a road, and it provides a simple interface to visualize the process.

## Features
- **Real-time Object Detection**: Uses the YOLO model to detect cars in each frame of the video.
- **Counting Mechanism**: A virtual line is placed in the middle of the road, and cars are counted as they cross this line.
- **Customizable Line Placement**: You can adjust the position of the line to count cars at different locations.
- **PyTorch Integration**: Leverages the power of PyTorch for running YOLO.
  
## Installation

### Prerequisites
- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

### Setup Instructions

1. Clone this repository:
    ```bash
    git clone https://github.com/AlgoriusAlhammak/Car-Counter.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Car-Counter
    ```
3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. To start counting cars from a video, run the following command:
    ```bash
    python Car-Count.py --input path/to/your/video.mp4 or simply hit run
    ```

2. The script will open a window displaying the video feed with detected cars and the counting line.

3. The detected cars will be marked with bounding boxes, and a counter will keep track of the number of cars crossing the line.

### Adjusting the Line Position

You can adjust the position of the counting line by modifying the `limits` variable in the `car_count.py` script:
```python
limits = [x1, y1, x2, y2]
These variables represent the start and end points of the line
```
## Demo
A sample video is included in the Demo/ folder to demonstrate the functionality of the system.

## Acknowledgements
1. Thanks to cvzone for contributing to the development of computer vision tools.

