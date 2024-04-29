#!/bin/bash

# Contact: https://github.com/Nerfs-Eye-View/nerf-project

###############################################################################
#                            Available Combinations                           #
###############################################################################

echo "┌───────┬───────────┬─────┬──────────────────────┐"
echo "│ NO.   │ FOV       │ FPS │ COLMAP mapper type   │"
echo "├───────┼───────────┼─────┼──────────────────────┤"
echo "│ 1     │ 130°      │ 0.5 │ mapper               │"
echo "│ 2     │ 130°      │ 0.5 │ hierarchical mapper  │"
echo "│ 3     │ 130°      │ 2   │ mapper               │"
echo "│ 4     │ 130°      │ 2   │ hierarchical mapper  │"
echo "│ 5     │ 130°      │ 10  │ mapper               │"
echo "│ 6     │ 130°      │ 10  │ hierarchical mapper  │"
echo "│ 7     │ 85°       │ 0.5 │ mapper               │"
echo "│ 8     │ 85°       │ 0.5 │ hierarchical mapper  │"
echo "│ 9     │ 85°       │ 2   │ mapper               │"
echo "│ 10    │ 85°       │ 2   │ hierarchical mapper  │"
echo "│ 11    │ 85°       │ 10  │ mapper               │"
echo "│ 12    │ 85°       │ 10  │ hierarchical mapper  │"
echo "│ 13    │ record3d  │ 0.5 │ X                    │"
echo "│ 14    │ record3d  │ 2   │ X                    │"
echo "│ 15    │ record3d  │ 10  │ X                    │"
echo "└───────┴───────────┴─────┴──────────────────────┘"

###############################################################################
#                                Download Packages                            #
###############################################################################

echo -n "Please enter the combination number you want to execute: "
read combination_number

# 필요한 패키지 다운로드(apt)
apt install -y zip unzip imagemagick git
echo -e '2\n133\n' apt install -y libopencv-dev python3-opencv

# 필요한 패키지 다운로드(pip)
pip install gdown tqdm torch

###############################################################################
#                                  Set Variables                              #
###############################################################################

video_fovs=("wide" "normal" "record3d")

declare -A gdrive_ids
gdrive_ids["wide"]="1dPblORqwimPgy-a8OFOeGfiYU79ArDNd"  # set id in google-drive url for video
gdrive_ids["normal"]=""
gdrive_ids["record3d"]=""

###############################################################################
#                                Define Functions                             #
###############################################################################

function execute_combination() {
	# execute_combination <combination number> <fov> <fps> <COLMAP command>
	local session_name="Comb_$1"
	local fov=$2
	local gdrive_id=${gdrive_ids[${fov}]}
	local fps=$3
	local basedir="/root"
	local workdir="$basedir/$session_name"
	local is_hierarchical=$4

	echo session_name: $session_name
	echo fov: $fov
	echo gdrive_id: $gdrive_id
	echo fps: $fps
	echo basedir: $basedir
	echo workdir: $workdir
	echo is_hierarchical: $is_hierarchical

	##############################################################
	#                         Preprocess                         #
	##############################################################

	tmux new-session -d -s "$session_name"

    # 명령어 로깅
    tmux pipe-pane -t "$session_name" "cat > ${workdir}/${session_name}.log"

	echo "Session $session_name has started. Output is being logged to ${session_name}.log"
	echo "You can attach to this session with 'tmux attach -t $session_name'."

    # tmux 세션에 명령어 전송
    tmux send-keys -t "$session_name" "echo Running combination $session_name" C-m
    tmux send-keys -t "$session_name" "sleep 2" C-m
    tmux send-keys -t "$session_name" "mkdir -p $workdir && cd $workdir" C-m
    tmux send-keys -t "$session_name" "mkdir -p $workdir/images" C-m
    tmux send-keys -t "$session_name" "gdown ${gdrive_id}" C-m
    tmux send-keys -t "$session_name" "unzip $fov" C-m
    tmux send-keys -t "$session_name" "alias python=python3" C-m
    tmux send-keys -t "$session_name" "python $basedir/nerf-project/utils/preprocess/multiple_video_sample_tqdm_jpg.py $fps $workdir/images $workdir/${fov}/*" C-m

	# Record3D는 COLMAP와 LLFF 수행 X
	if [[ $session_name -gt 12 ]]; then
		exit 0
	fi

	##############################################################
	#                           COLMAP                           #
	##############################################################

	# prepare
    tmux send-keys -t "$session_name" "echo Running combination $session_name" C-m
    tmux send-keys -t "$session_name" "sleep 2" C-m
    tmux send-keys -t "$session_name" "cd $workdir" C-m
    tmux send-keys -t "$session_name" "bash $basedir/nerf-project/utils/colmap/download_vocab_tree.sh" C-m
    tmux send-keys -t "$session_name" "mv $workdir/vocab_tree/*.bin $workdir" C-m

	# feature_extractor
    tmux send-keys -t "$session_name" "colmap feature_extractor --database_path ./database.db --image_path ./images" C-m

	# feature matcher
	local image_num=$(ls $workdir/images | wc -l)
	if [[ $image_num -lt 10000 ]]; then
		vocab_tree_bin='vocab_tree_flickr100K_words256K.bin'
	else
		vocab_tree_bin='vocab_tree_flickr100K_words1M.bin'
	fi
    tmux send-keys -t "$session_name" "colmap vocab_tree_matcher --database_path ./database.db --VocabTreeMatching.vocab_tree_path $vocab_tree_bin --SiftMatching.use_gpu 1" C-m

	# mapper
    tmux send-keys -t "$session_name" "mkdir -p $workdir/sparse" C-m
	if [[ $is_hierarchical -eq 0 ]]; then
		tmux send-keys -t "$session_name" "colmap mapper --database_path ./database.db --image_path ./images --output_path ./sparse --Mapper.ba_global_function_tolerance=0.000001" C-m
	else
		tmux send-keys -t "$session_name" "colmap hierarchical_mapper --database_path ./database.db --image_path ./images --output_path ./sparse --image_overlap 100" C-m
	fi


	##############################################################
	#                            LLFF                            #
	##############################################################

	# prepare
    tmux send-keys -t "$session_name" "echo Running combination $session_name" C-m
    tmux send-keys -t "$session_name" "sleep 2" C-m
    tmux send-keys -t "$session_name" "cd $workdir" C-m
    tmux send-keys -t "$session_name" "pip install -r $basedir/nerf-project/utils/llff/requirements.txt" C-m
    tmux send-keys -t "$session_name" "python $basedir/nerf-project/utils/llff/colmap2poses.py --project_path $workdir/ --model_path sparse/0" C-m
}


###############################################################################
#                            Run Specific Combination                         #
###############################################################################

case "$combination_number" in
    1)
        echo "Executing combination 1..."
		echo "│ 1     │ 130°      │ 0.5 │ mapper               │"
		execute_combination 1 "wide" "0.5" 0
        ;;
    2)
        echo "Executing combination 2..."
		echo "│ 2     │ 130°      │ 0.5 │ hierarchical mapper  │"
		execute_combination 2 "wide" "0.5" 1
        ;;
    3)
        echo "Executing combination 3..."
		echo "│ 3     │ 130°      │ 2   │ mapper               │"
		execute_combination 3 "wide" "2" 0
        ;;
    4)
        echo "Executing combination 4..."
		echo "│ 4     │ 130°      │ 2   │ hierarchical mapper  │"
		execute_combination 4 "wide" "2" 1
        ;;
    5)
        echo "Executing combination 5..."
		echo "│ 5     │ 130°      │ 10  │ mapper               │"
		execute_combination 5 "wide" "10" 0
        ;;
    6)
        echo "Executing combination 6..."
		echo "│ 6     │ 130°      │ 10  │ hierarchical mapper  │"
		execute_combination 6 "wide" "10" 1
        ;;
    7)
        echo "Executing combination 7..."
		echo "│ 7     │ 85°       │ 0.5 │ mapper               │"
		execute_combination 7 "normal" "0.5" 0
        ;;

    8)
        echo "Executing combination 8..."
		echo "│ 8     │ 85°       │ 0.5 │ hierarchical mapper  │"
		execute_combination 8 "normal" "0.5" 1
        ;;
    9)
        echo "Executing combination 9..."
		echo "│ 9     │ 85°       │ 2   │ mapper               │"
		execute_combination 9 "normal" "2" 0
        ;;
    10)
        echo "Executing combination 10..."
		echo "│ 10    │ 85°       │ 2   │ hierarchical mapper  │"
		execute_combination 10 "normal" "2" 1
        ;;
    11)
        echo "Executing combination 11..."
		echo "│ 11    │ 85°       │ 10  │ mapper               │"
		execute_combination 11 "normal" "10" 0
        ;;
    12)
        echo "Executing combination 12..."
		echo "│ 12    │ 85°       │ 10  │ hierarchical mapper  │"
		execute_combination 12 "normal" "10" 1
        ;;
    13)
        echo "Executing combination 13..."
		echo "│ 13    │ record3d  │ 0.5 │ X                    │"
		execute_combination 13 "record3d" "0.5" 1
        ;;
    14)
        echo "Executing combination 14..."
		echo "│ 14    │ record3d  │ 2   │ X                    │"
		execute_combination 14 "record3d" "2" 1
        ;;
    15)
        echo "Executing combination 15..."
		echo "│ 15    │ record3d  │ 10  │ X                    │"
		execute_combination 15 "record3d" "10" 1
        ;;

    *)
        exit 2
        ;;
esac
