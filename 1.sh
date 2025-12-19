#!/bin/bash

# 定义输入文件夹变量
INPUT_DIR="./images/2025-06-17/down"

# 运行原始命令
python amg.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input "$INPUT_DIR" --output ./output

# 获取输入文件夹的基本名称
INPUT_BASENAME=$(basename "$INPUT_DIR")

# 将output文件夹打包成zip文件，以输入文件夹命名
zip -r "${INPUT_BASENAME}.zip" ./output

rm -rf ./output
mkdir output
