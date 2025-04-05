# Bottle-detection AI

This project is a real-time AI system that detects **when a person throws a bottle** using computer vision. It combines **YOLO object detection** with tracking and distance logic to identify when a bottle held by a person is moved far away — indicating a potential "throw".

When a throw is detected:
- 📸 A screenshot is saved
- 🎞️ A composite video is generated (screenshot + random reaction clip)
- ▶️ The video is optionally auto-played using VLC Media Player

---

## 🚀 Features

- ✅ Real-time detection with YOLOv8 or YOLOv12
- 🧠 Smart throw detection using distance tracking
- 🎥 Composite video generation with audio
- 📦 Automatic saving of detection screenshots and videos
- 🖥️ Multiple command-line modes

---

## 🛠️ Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bottle-throw-detection.git
   cd bottle-throw-detection
   
2. **Install dependencies:**
  pip install -r requirements.txt

Note: Make sure VLC Media Player is installed and update the vlc_path variable in the script if needed.

3. **Run the detection system with Python using the available command-line flags:**
py -m video_detection_feedback_video_args [flags]

✅ Available Flags
   -a	Do everything: show webcam, detect, create & play video
   -wc	Show only the webcam feed
   -v	Create a reaction video when a throw is detected
   -pv	Play the video after it's created

Example Commands
Full workflow (detect + record + play):
py -m video_detection_feedback_video_args -a

Webcam feed only:
py -m video_detection_feedback_video_args -wc

Create but don’t play video:
py -m video_detection_feedback_video_args -v

3. **Folder Structure**
📂 detections/
  ├─ detection_*.jpg     # Screenshots of throw detections
  ├─ output_*.mp4        # Reaction videos
📂 TransVideo/
  ├─ *.mp4               # Source reaction clips
📄 video_detection_feedback_video_args.py  # Main script
📜 License
This project uses a modified version of the MIT License.

⚠️ This software is licensed for non-commercial, open-source, and educational use only.
Commercial use is not permitted without written permission from the author.

See the LICENSE file for full terms.

🤝 Contributing
Contributions are welcome!
If you'd like to improve detection, add new features, or help optimize the code, feel free to fork the repo and submit a pull request.

📬 Contact
For questions or collaboration inquiries, contact 0-ToniK-0 on GitHub.
