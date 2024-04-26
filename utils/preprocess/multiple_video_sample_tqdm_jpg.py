import cv2
import os
import sys
import shutil
from tqdm import tqdm

# Global image counter
image_counter = 1

def extract_frames(video_path, desired_fps, output_path, global_image_counter):
    global image_counter  # Use the global image counter

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return  # Return to allow processing of next video

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(video_fps // desired_fps))  # Calculate step

    print(f"Processing Frames for {os.path.basename(video_path)}...")

    # Setup tqdm progress bar
    pbar = tqdm(total=total_frames // step, desc="Extracting Frames")

    current_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame % step == 0:
            filename = os.path.join(output_path, f'{image_counter:06d}.jpg')
            # Save the frame as JPEG file
            cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Set JPEG quality
            image_counter += 1  # Increment global counter
            pbar.update(1)  # Update progress bar

        current_frame += 1

    cap.release()
    pbar.close()  # Ensure progress bar closes properly
    sys.stdout.write(f"\nComplete! Images have been saved to '{output_path}' folder.\n")
    return image_counter

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <desired_fps> <output_path> <video_file_path_1> <video_file_path_2> ...")
        sys.exit()

    desired_fps = float(sys.argv[1])
    output_path = sys.argv[2]
    video_paths = sys.argv[3:]

    for video_path in video_paths:
        image_counter = extract_frames(video_path, desired_fps, output_path, image_counter)

