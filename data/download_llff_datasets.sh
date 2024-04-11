#!/bin/bash

set -xe

# make llff
if [[ ! -d llff ]]; then
	mkdir ./llff
fi

###################################[ download command ]###################################

download() {
	curl -L "https://drive.usercontent.google.com/download?id=$1&confirm=xxx" -o $2
}

# 파일 ID와 이름 쌍의 리스트
declare -a dataset_files=(
    "1duCAX76-pTLRGHW8EzaWGYOZwPfxXsqj koncow"
    # 추가 파일 쌍
    #"file_id_konlibrary konlibrary.zip"
)

# 파일 리스트를 순회하면서 다운로드
for file_info in "${dataset_files[@]}"; do
    IFS=' ' read -r file_id file_name <<< "$file_info"
    download "$file_id" "$file_name.zip"
    
    # 파일 유형에 따라 적절한 처리 실행
	unzip "$file_name.zip" -d ./llff
	rm "$file_name.zip" # 압축 해제 후 원본 zip 파일 삭제
done

echo "All files have been processed successfully."


