#!/bin/bash

set -xe

if [[ $# -ne 2 ]]; then
	echo "Usage: remove_not_included.sh <image folder> <image list file>"
	exit 1
fi

ls $1 | cat > images_list.txt
echo >> $2
diff images_list.txt $2 | grep '<' | tr -d '<' | xargs -I{} rm $1/{}
rm images_list.txt
