# ⚽ YOLO Football Detector

[![Python](https://img.shields.io/badge/Python–3.8%2B-blue.svg)](https://www.python.org/downloads/)  
[![OpenCV](https://img.shields.io/badge/OpenCV–4.6.0%2B-green.svg)]()  
[![YOLOv8](https://img.shields.io/badge/YOLOv8–8.x-red.svg)](https://github.com/ultralytics/ultralytics)  
[![License](https://img.shields.io/badge/License–MIT-yellow.svg)](LICENSE)

A lightweight project to detect football players and the ball in video using YOLOv8 and K-means color clustering for basic team differentiation.

---

## 🎥 Demo

https://github.com/user-attachments/assets/be5017b9-90bc-4288-b32e-9b0fb12a0a82

---

## 🚀 Features

- Player and ball detection using **YOLOv8**
- **Basic color clustering** (K‑means) to distinguish teams by jersey color
- Annotated output video generation
- Configurable parameters (confidence, video path, weights)
- Supports CPU and GPU (CUDA)

---

## 🧩 What It Actually Does

1. Loads a YOLOv8 model (e.g., `best.pt`)
2. Processes each frame of an input video
3. Runs object detection and filters relevant classes
4. Applies color clustering to assign team labels
5. Draws bounding boxes and saves the result video to `output/`

---

## 📋 Quick Start

### Requirements

- Python 3.8+
- CUDA GPU (optional)

### Setup

```bash
git clone https://github.com/Shehab-Hegab/YOLO-Football-Detector.git
cd YOLO-Football-Detector
pip install -r requirements.txt
```

Place the YOLOv8 weights (e.g. `best.pt`) inside the `weights/` folder.

### Run

```bash
python main.py --source test_videos/CityUtdR.mp4 --weights weights/best.pt --conf 0.5
```

| Argument     | Description                     | Default                      |
|--------------|----------------------------------|------------------------------|
| `--source`   | Path to input video             | `test_videos/CityUtdR.mp4`   |
| `--weights`  | Path to YOLOv8 model weights    | `weights/best.pt`            |
| `--conf`     | Detection confidence threshold  | `0.5`                        |

Output is saved as `output/<video_name>_out.mp4`.

---

## 🗂️ Project Structure

```
.
├── main.py
├── requirements.txt
├── weights/
│   ├── best.pt
│   └── last.pt
├── test_videos/
│   └── CityUtdR.mp4
└── output/
    └── CityUtdR_out.mp4
```

---

## 🧠 How It Works

### Detection

YOLOv8 is used to detect:
- Players
- Ball

### Team Labeling (Basic)

1. Extracts color information from each player bounding box.
2. Applies K-means clustering with `n_clusters=2` to group players into two teams.
3. Assigns a label (e.g., "Team A" or "Team B") for annotation.

### Output

Each frame is annotated with:
- Bounding boxes
- Class labels (Player/Ball)
- Color-coded teams

The processed video is saved to the `output/` directory.

---

## 🔧 Configuration

Edit `main.py` to change:

```python
conf_threshold = 0.5  # Detection threshold
n_clusters = 2        # Number of teams (for K-means)
```

Adjust bounding box colors or labels in:
```python
box_colors = {
    "player_team_0": (255, 0, 0),
    "player_team_1": (0, 255, 0),
    "ball": (0, 0, 255),
}
```

---

## ✅ Code Review

- ✔️ Detection and annotation working correctly
- ⚠️ Team classification may be inaccurate in cases of similar jersey colors or poor lighting
- 🟡 Referees, goalkeepers, or staff are **not detected**
- 🟣 No real-time display (output saved directly to video)

---

## 📚 Dependencies

- `ultralytics`
- `opencv-python`
- `numpy`
- `scikit-learn`

Install them via:

```bash
pip install -r requirements.txt
```

---

## 📈 Improvement Ideas

- Add real-time display using OpenCV
- Improve team classification using more advanced color features
- Add support for referee/goalkeeper detection by retraining the model
- Include player tracking across frames

---

## 🤝 Contributing

Pull requests are welcome! Fork the repo and suggest improvements.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

## 📞 Contact

For suggestions or help, please open an issue on GitHub.

---

**Built with ❤️ for football AI experiments**
