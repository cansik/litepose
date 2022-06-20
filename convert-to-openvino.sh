#!/usr/bin/env bash

# install dependencies
pip install openvino-dev

precisions=("fp32" "fp16")

for file in $(ls models/*.onnx)
do
  file_base="${file%.*}"
  echo "converting $file_base..."

  for precision in ${precisions[@]}; do
    mo --input_model "$file" --batch 1 --mean_values "[124, 116, 104]" --scale 255 --reverse_input_channels --data_type ${precision^^} --output_dir "$file_base-$precision"
  done
done

echo "done"