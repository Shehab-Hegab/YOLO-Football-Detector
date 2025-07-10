# ⚽ Football Object Detection


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.168+-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated computer vision system that automatically detects and classifies objects in football match videos using YOLOv8 and advanced color-based team classification algorithms.

## 🎯 Features

- **Multi-Object Detection**: Detects players, goalkeepers, ball, referees, and staff
- **Automatic Team Classification**: Uses K-means clustering to separate players into teams based on kit colors
- **Smart Team Positioning**: Automatically determines which team is on the left vs right side
- **Grass Color Detection**: Adapts to different pitch conditions and lighting
- **Real-time Processing**: Processes video frames with progress tracking
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📊 Detection Classes

| Class | Description | Color Code |
|-------|-------------|------------|
| 0 | Player (Team Left) | 🔴 Red |
| 1 | Player (Team Right) | 🔵 Blue |
| 2 | Goalkeeper (Team Left) | 🟢 Green |
| 3 | Goalkeeper (Team Right) | 🟡 Yellow |
| 4 | Ball | 🟣 Purple |
| 5 | Main Referee | 🔵 Light Blue |
| 6 | Side Referee | 🟣 Pink |
| 7 | Staff Members | ⚫ Black |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Football-Object-Detection.git
   cd Football-Object-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained weights** (if not included)
   ```bash
   # Weights should be in the weights/ directory
   # - best.pt (50MB)
   # - last.pt (50MB)
   ```

### Usage

#### Basic Usage
```bash
# Process default test video
python main.py

# Process custom video
python main.py "path/to/your/video.mp4"
```

#### Example
```bash
python main.py "test_videos/CityUtdR.mp4"
```

**Output**: Annotated video saved as `output/CityUtdR_out.mp4`

## 🧠 How It Works

### 1. Initial Frame Analysis
The system analyzes the first frame to establish baseline information:

#### Grass Color Detection
- Converts frame to HSV color space
- Filters green color ranges to identify pitch
- Calculates average grass color for background removal

#### Team Classification Setup
- Extracts player bounding boxes
- Removes grass background from each player
- Uses K-means clustering to group players by kit colors
- Determines team positioning (left vs right)

### 2. Frame-by-Frame Processing
For each video frame:

1. **YOLO Detection**: Runs YOLOv8 inference to detect objects
2. **Player Classification**: Classifies players into teams using pre-trained K-means model
3. **Team Assignment**: Assigns "Left" or "Right" labels based on first frame analysis
4. **Goalkeeper Positioning**: Classifies goalkeepers based on field position
5. **Annotation**: Draws bounding boxes and labels on frame

### 3. Advanced Algorithms

#### Color-Based Team Separation
```python
# K-means clustering for team classification
kits_kmeans = KMeans(n_clusters=2)
kits_kmeans.fit(kits_colors)
```

#### Grass Background Removal
```python
# HSV-based grass filtering
lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
```

## 📁 Project Structure

```
Football-Object-Detection/
├── main.py                           # Main processing script
├── Football_Object_Detection.ipynb   # Detailed analysis notebook
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
├── weights/                         # Trained YOLO models
│   ├── best.pt                     # Best model weights
│   └── last.pt                     # Latest model weights
├── test_videos/                     # Sample input videos
│   └── CityUtdR.mp4               # Test video
└── output/                         # Processed videos
    └── CityUtdR_out.mp4           # Annotated output
```

## 🔧 Configuration

### Model Parameters
- **Confidence Threshold**: 0.5 (adjustable in `main.py`)
- **Frame Rate**: 30 FPS (configurable)
- **Video Codec**: MP4V (changeable)

### Color Detection Settings
```python
# HSV color ranges for grass detection
lower_green = np.array([25, 40, 40])
upper_green = np.array([85, 255, 255])

# Team classification parameters
n_clusters = 2  # Number of teams
```

## 📈 Performance

- **Processing Speed**: ~30 FPS (depends on hardware)
- **Accuracy**: High precision for player detection and team classification
- **Memory Usage**: Optimized for large video files
- **GPU Acceleration**: CUDA support for faster processing

## 🛠️ Customization

### Adding New Object Classes
1. Retrain YOLO model with new classes
2. Update `labels` list in `main.py`
3. Add corresponding color codes to `box_colors`

### Modifying Team Classification
- Adjust K-means parameters in `get_kits_classifier()`
- Modify color filtering in `get_kits_colors()`
- Change team positioning logic in `get_left_team_label()`

## 🐛 Troubleshooting

### Common Issues

**Video not processing:**
- Check video format (MP4 recommended)
- Ensure video file exists and is not corrupted
- Verify all dependencies are installed

**No output generated:**
- Check `output/` directory permissions
- Ensure sufficient disk space
- Verify video codec compatibility

**Poor detection accuracy:**
- Adjust confidence threshold in `main.py`
- Check lighting conditions in video
- Verify model weights are properly loaded

### Debug Mode
Enable debug prints by uncommenting progress tracking lines in `main.py`:
```python
print(f"Processing frame {frame_count}/{total_frames}")
```

## 📚 Technical Details

### Model Architecture
- **Base Model**: YOLOv8 (Ultralytics)
- **Training Dataset**: SoccerNet Dataset
- **Training Epochs**: 25
- **Input Resolution**: Variable (maintains aspect ratio)

### Dependencies
- `ultralytics==8.0.168` - YOLO model framework
- `opencv-python==4.6.0.66` - Computer vision operations
- `numpy==1.22.4` - Numerical computations
- `scikit-learn==1.3.0` - K-means clustering

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [SoccerNet Dataset](https://www.soccer-net.org/) for training data
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

If you encounter any issues or have questions:

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/Football-Object-Detection/issues)
- 📖 Documentation: Check the Jupyter notebook for detailed analysis

---

**Made with ❤️ for the football community**

https://github.com/user-attachments/assets/be5017b9-90bc-4288-b32e-9b0fb12a0a82
