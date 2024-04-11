import cv2
import sys

def resize_video(input_video_path, output_video_path, scale=0.3):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get original dimensions and fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Define the codec for .MOV format and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec adjusted for .MOV format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
    
    # Read and resize each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Write the resized frame to the output video
        out.write(resized_frame)
    
    # Release everything if job is finished
    cap.release()
    out.release()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py input_video_path output_video_path")
        sys.exit(1)
    
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    scale_factor = float(sys.argv[3])
    resize_video(input_video_path, output_video_path, scale_factor)

