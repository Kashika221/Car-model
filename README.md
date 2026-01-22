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

## Technology Stack

- **YOLOv4**: State-of-the-art object detection model
- **SORT**: Simple Online and Realtime Tracking algorithm
- **Python 3.x**: Core programming language
- **OpenCV**: Image and video processing
- **NumPy**: Numerical computations
- **TensorFlow/PyTorch**: Deep learning framework

### Detection Phase
1. Each frame is processed by YOLOv4 neural network
2. Car objects are detected with bounding boxes
3. Confidence scores filter low-quality detections

### Tracking Phase
1. SORT algorithm assigns unique IDs to detected cars
2. Tracks are maintained across consecutive frames
3. ID persistence ensures consistent counting

### Counting Phase
1. Virtual counting lines are defined in the video
2. Cars crossing the line are counted
3. Results are displayed and saved

## Output

The system generates:
- **Annotated Video**: Original video with bounding boxes, IDs, and car count
- **Count Log**: CSV file with frame-by-frame vehicle counts
- **Statistics**: Traffic flow analysis and density information

## Configuration

Customize the system by modifying parameters in `car counter.py`:

```python
# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Tracking parameters
MAX_AGE = 30  # Maximum frames to keep alive a track
MIN_HITS = 3  # Minimum detections before confirming track

# Counting line position
COUNTING_LINE_Y = 400  # Y-coordinate of counting line
```

## Performance Optimization

- Use GPU acceleration for faster processing
- Reduce frame resolution for real-time processing
- Adjust confidence thresholds based on accuracy requirements
- Use multi-threading for parallel frame processing

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| Python | 3.7 | 3.9+ |
| RAM | 4GB | 8GB+ |
| Storage | 500MB | 2GB+ |
| GPU | Optional | NVIDIA CUDA |

## Supported Formats

**Video Formats**: MP4, AVI, MOV, MKV, FLV

**Image Formats**: JPG, PNG, BMP

## Troubleshooting

### Issue: "YOLO weights not found"
**Solution**: Ensure YOLOv4 weights are in the `yolo weights/` directory with correct filenames.

### Issue: Poor detection accuracy
**Solution**: Adjust confidence threshold lower, improve video lighting, or use higher resolution input.

### Issue: Slow performance
**Solution**: Reduce input video resolution, skip frames, or enable GPU acceleration.

### Issue: Tracking ID flickering
**Solution**: Increase MIN_HITS parameter or adjust SORT confidence threshold.

## Future Enhancements

- Multi-directional counting (left-to-right, right-to-left)
- Vehicle classification by type (car, truck, bus)
- Lane-wise traffic analysis
- Speed estimation using optical flow
- Cloud-based processing support
- Web-based dashboard for monitoring

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- YOLOv4: Bochkovskiy et al. (https://github.com/AlexeyAB/darknet)
- SORT: Bewley et al. (https://arxiv.org/abs/1602.00763)
- COCO Dataset: Lin et al.

## References

- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [SORT Paper](https://arxiv.org/abs/1602.00763)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Darknet YOLOv4](https://github.com/AlexeyAB/darknet)

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review YOLOv4 and SORT algorithm papers

## Disclaimer

This project is designed for educational and research purposes. Users are responsible for compliance with local laws and regulations when deploying this system for traffic monitoring.

---

**Built with precision for intelligent traffic monitoring** 
