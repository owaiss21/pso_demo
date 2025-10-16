import cv2
from pathlib import Path
import time

def parse_time(timestr):
    """Parse a time string HH:MM:SS into seconds."""
    parts = [int(p) for p in timestr.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)  # pad with zeros if needed
    h, m, s = parts
    return h * 3600 + m * 60 + s

def crop_video(input_path, start_time, end_time):
    input_path = Path(input_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration (s): {duration}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = input_path.with_name(f"{input_path.stem}_crop2{input_path.suffix}")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    start_sec = parse_time(start_time)
    end_sec = parse_time(end_time)

    if start_sec >= duration:
        print("Start time is beyond video duration.")
        cap.release()
        out.release()
        return

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = min(start_frame, total_frames - 1)
    end_frame = min(end_frame, total_frames - 1)

    print(f"Cropping from frame {start_frame} to {end_frame} (seconds {start_sec} to {end_sec})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written_frames = 0
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped reading at frame {frame_idx}")
            break
        out.write(frame)
        written_frames += 1

    cap.release()
    out.release()
    print(f"Cropped video saved to {output_path}")
    print(f"Frames written: {written_frames}, Duration: {written_frames/fps:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop a video between start and end times (HH:MM:SS).")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("start_time", help="Start time in HH:MM:SS")
    parser.add_argument("end_time", help="End time in HH:MM:SS")
    args = parser.parse_args()
    
    start = time.time()
    crop_video(args.video_path, args.start_time, args.end_time)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")