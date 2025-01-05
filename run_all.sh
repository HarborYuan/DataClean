mkdir -p results

for folder in data/vid/0{72..81}*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        result_file="results/${folder_name}_result.txt"
        PYTHONUNBUFFERED=1 PYTHONPATH=. python test_script/rvos_sa2va_script.py --model_path OMG-Research/Sa2VA-8B --video-list-folder "$folder" | tee "$result_file"
        echo "Results for $folder saved to $result_file"
    fi
done
