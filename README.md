# Car Detection and Counting using YOLOv4 

This project demonstrates a real-time car detection and counting system utilizing the YOLOv4 object detection algorithm in conjunction with the SORT (Simple Online and Realtime Tracking) tracking algorithm. The system processes video streams to detect and count vehicles, making it suitable for applications like traffic monitoring and analysis.

## Repository Contents

* `car counter.py`: Main script that integrates YOLOv4 for object detection and SORT for tracking to count cars in video streams.
* `sort.py`: Implementation of the SORT tracking algorithm to maintain identities of detected objects across frames.
* `yolo weights/`: Directory containing the pre-trained YOLOv4 weights and configuration files.
* `images/`: Folder containing sample images or frames used for testing and demonstration purposes.

## Prerequisites

* Python 3.x
* OpenCV
* NumPy
* TensorFlow or PyTorch (depending on the YOLOv4 implementation used)

*Note: Ensure that the YOLOv4 weights and configuration files are correctly placed in the `yolo weights/` directory.*

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kashika221/Car-model.git
   cd Car-model
   ```

3. **Run the car counting script:**

   ```bash
   python car\ counter.py
   ```

   *Ensure that the video source is correctly specified within the script.*
## Features

* **Real-time Detection:** Utilizes YOLOv4 for fast and accurate object detection.
* **Object Tracking:** Implements SORT to track multiple vehicles across frames.
* **Vehicle Counting:** Counts the number of vehicles passing through a defined region in the video.

## Sample Output
