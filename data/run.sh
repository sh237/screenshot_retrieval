folder_path=$(pwd)
subfolders=$(find "$folder_path" -type d)
for subfolder in $subfolders; do
    png_files=$(find "$subfolder" -maxdepth 1 -type f -iname "*.png")
    file_count=$(echo "$png_files" | wc -l)
    if [ $file_count -eq 1 ]; then
        png_filename=$(basename "$png_files")
        tesseract "$subfolder/$png_filename" $subfolder/raw_transcription -l eng
    else
        echo "Error: Each folder must have 1 PNG file."
    fi
done

