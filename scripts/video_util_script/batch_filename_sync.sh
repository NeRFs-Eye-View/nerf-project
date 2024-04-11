#!/bin/bash

set -e

cd /Users/jungcow/Documents/학사/24-1/nerfs-eye-view/video_util_script

prev_num_of_images=0
offset=1
for i in {1..8}; do
    num_of_images=$(ls ./konlibrary$i/*.png | wc -l);
    for j in $(seq $offset $(($num_of_images + $offset - 1))); do
        mv ./konlibrary$i/$(printf "000%03d.png" "$j") ./konlibrary$i/$(printf "000%03d.png" $(($j + $prev_num_of_images - $offset - 1)));
    done
    prev_num_of_images=$prev_num_of_images+$num_of_images
done