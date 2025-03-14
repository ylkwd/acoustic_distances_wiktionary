#!/bin/bash

dirs=(
    "../datasets/wiktionary_pronunciations-final/audios/GPT4o"
    "../datasets/wiktionary_pronunciations-final/audios/wiktionary"
)

for input_dir in "${dirs[@]}"; do
    echo "Processing directory: $input_dir"
    for file in "$input_dir"/*.wav; do
        tmp_file="${file%.wav}.tmp.wav"
        echo "Processing $file..."
        if sox "$file" -r 16000 -c 1 -b 16 "$tmp_file"; then
            mv "$tmp_file" "$file"
            echo "Successfully re-encoded $file"
        else
            echo "Error processing $file"
            rm -f "$tmp_file"
        fi
    done
done
