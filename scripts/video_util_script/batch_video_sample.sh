#!/bin/bash

set -xe


for i in {1..8};
    do python multiple_video_sample.py '0.5' ./konlibrary$i ../nerf-data/raw_video/kon_library/konlibrary0$i.MOV;
done
