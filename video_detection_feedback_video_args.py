import cv2
import os
import datetime
import math
from ultralytics import YOLO
import pyautogui
import time
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
import os
import random
import glob
import subprocess
import argparse

# CLI options
parser = argparse.ArgumentParser(
    description="Bottle Detection System — Capture, Generate & Play Reaction Video",
    epilog="""
Example Usages:
  py -m video_detection_feedback_video_v2 -a       # Do everything
  py -m video_detection_feedback_video_v2 -v       # Only create video
  py -m video_detection_feedback_video_v2 -pv      # Only play video and create video
  py -m video_detection_feedback_video_v2 -wc      # Show webcam feed (no video)

Flags can be combined: 
  -v -pv   → Create and play video
  -a       → Shortcut for -v -pv -wc

To exit the program, press 'q'.

If this is your firtst time running the script, make sure to install the required packages:
  pip install -r requirements.txt
If you are getting a vlc error make sure to change the vlc_path variable to the correct path of your vlc installation.
Do not use a different video player, as it may not support the same features.
"""
)
parser.add_argument('-pv', action='store_true', help='Play the video after it is created')
parser.add_argument('-v', action='store_true', help='Create the output video')
parser.add_argument('-wc', action='store_true', help='Show the webcam feed')
parser.add_argument('-a', action='store_true', help='Do everything: show cam + create + play')
args = parser.parse_args()

# Determine flags
play_video = args.pv or args.a
create_video = args.v or args.a or args.pv
show_cam = args.wc or args.a

# Load YOLOv12 model with tracking
model = YOLO("yolov10s.pt")

# Create folder to save logs and screenshots
if not os.path.exists("detections"):
    os.makedirs("detections")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # or 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # or 1080



# Get screen size
screen_width, screen_height = pyautogui.size()
usable_height = screen_height - 150
vlc_path = r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"  # Adjust if installed elsewhere

if show_cam:
    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detection", screen_width, usable_height)

# To store previous distances between person and bottle
previous_distances = {}  # key: (person_id, bottle_id), value: distance
NEAR_DISTANCE = 300      # Considered "holding" or "about to throw"
FAR_DISTANCE = 500       # Considered "thrown"
i = 0
last_screenshot_time = 0  # Time of the last screenshot

def create_final_video_with_audio(screenshot_frame, video_path, output_path, video_width=760, video_height=80):
    # Resize screenshot to match video width
    sh, sw = screenshot_frame.shape[:2]
    resized_screenshot = cv2.resize(screenshot_frame, (video_width, int(sh * video_width / sw)))
    screenshot_path = "detections/tmp_screenshot.jpg"
    cv2.imwrite(screenshot_path, resized_screenshot)

    # Load video and image
    video_clip = VideoFileClip(video_path)
    screenshot_clip = ImageClip(screenshot_path).with_duration(video_clip.duration)

    # Resize video clip
    video_clip = video_clip.resized(height=video_height, width=video_width)

    # Set position: screenshot on top, video below
    final_height = resized_screenshot.shape[0] + video_height
    composite = CompositeVideoClip([
        screenshot_clip.with_position(("center", 0)),
        video_clip.with_position(("center", resized_screenshot.shape[0]))
    ], size=(video_width, final_height))

    # Keep audio and export final video
    composite = composite.with_audio(video_clip.audio)
    composite.write_videofile(output_path, codec="libx264", audio_codec="aac")

def create_composite_video(screenshot_frame, video_path, output_path, duration_sec=2, video_width=1200, video_height=80):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * duration_sec)

    # Resize screenshot to match video width
    sh, sw = screenshot_frame.shape[:2]
    resized_screenshot = cv2.resize(screenshot_frame, (video_width, int(sh * video_width / sw)))

    # Output video height = screenshot height + video height
    output_size = (video_width, resized_screenshot.shape[0] + video_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    count = 0
    while count < total_frames:
        ret, vid_frame = cap.read()
        if not ret:
            break
        vid_frame = cv2.resize(vid_frame, (video_width, video_height))

        composite_frame = cv2.vconcat([resized_screenshot, vid_frame])
        out.write(composite_frame)
        count += 1

    cap.release()
    out.release()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, verbose=False)

    bottle_data = []  # [(id, center)]
    person_data = []  # [(id, center)]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            conf = box.conf[0].item()
            track_id = int(box.id[0]) if box.id is not None else -1

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)

            if label == "bottle":
                bottle_data.append((track_id, center))
                color = (0, 255, 0)
            elif label == "person":
                person_data.append((track_id, center))
                color = (255, 0, 0)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for b_id, b_center in bottle_data:
        for p_id, p_center in person_data:
            key = (p_id, b_id)
            dist = math.hypot(b_center[0] - p_center[0], b_center[1] - p_center[1])

            # Line color: red if too far, yellow otherwise
            line_color = (0, 0, 255) if dist > FAR_DISTANCE else (0, 255, 255)

            # Draw line
            cv2.line(frame, b_center, p_center, line_color, 2)

            # Show distance text
            mid_x = (b_center[0] + p_center[0]) // 2
            mid_y = (b_center[1] + p_center[1]) // 2
            cv2.putText(frame, f"{int(dist)} px", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

            # Check previous state
            prev_dist = previous_distances.get(key, None)

            # If previously close and now far → detection event
            if prev_dist is not None and prev_dist > NEAR_DISTANCE and dist < FAR_DISTANCE:
                current_time = time.time()
                if current_time - last_screenshot_time >= 1:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    img_filename = f"detections/detection_{timestamp}.jpg"
                    log_filename = "detections/detection_log.txt"
                    # Save screenshot
                    cv2.imwrite(img_filename, frame)

                    if create_video:
                        # Create composite video
                        video_output = f"detections/output_{timestamp}.mp4"
                        # Pick a random video from TransVideo folder
                        video_choices = glob.glob("TransVideo/*.mp4")
                        if video_choices:
                            random_video = random.choice(video_choices)
                            create_final_video_with_audio(frame, random_video, video_output)
                            print(f"[VIDEO USED] {random_video}")
                        else:
                            print("[WARNING] No videos found in TransVideo/ folder!")
                        print(f"[VIDEO CREATED] {video_output}")
                        if play_video:
                            # Automatically open the video after it's created
                            try:
                                abs_video_path = os.path.abspath(video_output)
                                # Play with VLC and auto-close after playback

                                try:
                                    subprocess.Popen([
                                        vlc_path,
                                        abs_video_path,
                                        "--play-and-exit",
                                        "--fullscreen",     # auto-close after playback
                                        "--quiet",             # no messages
                                        "--no-video-title-show"
                                    ])
                                    print(f"[PLAYING WITH VLC] {abs_video_path}")
                                except Exception as e:
                                    print(f"[ERROR] Couldn't play with VLC: {e}")

                            except Exception as e:
                                print(f"[ERROR] Couldn't open video: {e}")

                    with open(log_filename, "a") as log_file:
                        log_file.write(f"[{timestamp}] THROW DETECTED between person {p_id} and bottle {b_id}. Distance changed from {prev_dist:.1f} to {dist:.1f} px\n")

                    print(f"[LOGGED] Throw detected: {img_filename} " + str(i))
                    i += 1
                    last_screenshot_time = current_time  # ⏱️ Update timer

            # Update the distance
            previous_distances[key] = dist

    if show_cam:
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
