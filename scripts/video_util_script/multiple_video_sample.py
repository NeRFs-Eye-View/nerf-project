import cv2
import os
import sys
import shutil

def print_progress_bar(percentage, total_width):
    progress_width = total_width - 10  # Adjust length for percentage numbers and padding
    filled_length = int(progress_width * percentage // 100)
    bar = '=' * filled_length + '-' * (progress_width - filled_length)
    sys.stdout.write(f'\r[{bar}] {percentage:.2f}%')  # Move cursor to the beginning of the line
    sys.stdout.flush()

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

    current_frame = 0

    print(f"Processing Frames for {os.path.basename(video_path)}...")
    sys.stdout.write("\033[s")  # Save cursor position

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame % step == 0:
            filename = os.path.join(output_path, f'{image_counter:06d}.png')
            cv2.imwrite(filename, frame)
            image_counter += 1  # Increment global counter
            percentage = (current_frame / total_frames) * 100
            columns, _ = shutil.get_terminal_size()
            sys.stdout.write("\033[u")  # Restore cursor position
            print_progress_bar(percentage, columns)

        current_frame += 1

    cap.release()
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

