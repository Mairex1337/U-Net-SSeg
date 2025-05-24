#!/bin/bash

# Note: need same venv with requirements and location as specified here 
#       for it to work.
# Run with: ./scripts/colorize_mask {input_dir} {output_dir}
source ../bdd100k/bddenv/bin/activate

input_dir="$(pwd)/${1}"
output_dir="$(pwd)/${2}"

mkdir -p "$output_dir"

cd ../bdd100k

python -m bdd100k.label.to_color -m sem_seg -i "$input_dir" -o "$output_dir"