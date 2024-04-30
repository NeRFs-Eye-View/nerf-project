import os
import sys

def find_missing_images(image_dir, text_file_path):
    # 이미지 파일 리스트를 읽어온다.
    images_set = set(os.listdir(image_dir))
    
    # 텍스트 파일에서 이미지 파일 이름을 읽어온다.
    with open(text_file_path, 'r') as file:
        listed_images = set(file.read().split())

    # 이미지 디렉토리에는 있지만, 텍스트 파일에는 없는 파일을 찾는다.
    missing_in_text = images_set - listed_images
    
    return missing_in_text

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <text_file_path>")
        sys.exit(1)

    # 사용자가 지정한 디렉토리와 텍스트 파일의 경로를 가져온다.
    image_directory = sys.argv[1]
    text_file = sys.argv[2]

    # 누락된 이미지 파일 찾기
    missing_images = find_missing_images(image_directory, text_file)

    # 결과 출력
    print("텍스트 파일에 빠진 이미지 파일들:")
    for img in sorted(missing_images):
        print(img)
