#!/bin/bash

folders=$(find data/benchmark -maxdepth 1 -type d -name "0*" | awk -F'/' '{name=$NF; if (substr(name, 1, 3) >= "072") print}' | sort)
if [ -z "$folders" ]; then
    echo "No folders starting with 0 found in data/benchmark."
    exit 1
fi

echo "Found the following folders:"
echo "$folders"

MODEL="ByteDance/Sa2VA-26B"

set -x

for folder in $folders; do
    echo "Running script for folder: $folder"
    PYTHONPATH=. python test_script/rvos_sa2va_script.py --model_path "$MODEL" --video-list-folder "$folder"
    if [ $? -ne 0 ]; then
        echo "Error running script for folder: $folder"
        exit 1
    fi
done

echo "All folders processed successfully."
