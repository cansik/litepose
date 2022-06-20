#!/usr/bin/env bash

# install dependencies
pip install openvino-dev

for file in $(ls "models/*.onnx")
do
  file_base = "${file%.*}"
  echo "$file_base"
done

# python -m mo --input_model unet.pdmodel --mean_values [0.485, 0.456, 0.406] --scale [229, 224, 225] --output_dir "models/"