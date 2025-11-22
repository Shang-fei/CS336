#!/bin/bash
# 脚本功能：使用国内 Hugging Face 镜像（hf-mirror.com）加速下载 CS336 课程所需的数据集。

# set -e: 确保任何命令失败时脚本立即退出，防止错误继续传播。
set -e

echo "--- 1. 创建 'data' 文件夹并进入 ---"
mkdir -p data
cd data

# --- TinyStories 数据集（使用镜像加速） ---
echo "--- 2. 开始下载 TinyStories V2 训练集 (train.txt) ---"
# 原始链接: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

echo "--- 3. 开始下载 TinyStories V2 验证集 (valid.txt) ---"
# 原始链接: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# --- OWT-Sample 数据集（使用镜像加速并解压） ---
echo "--- 4. 开始下载 OWT-Sample 训练集 (train.txt.gz) ---"
# 原始链接: https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz

echo "--- 5. 正在解压 OWT-Sample 训练集... ---"
gunzip owt_train.txt.gz

echo "--- 6. 开始下载 OWT-Sample 验证集 (valid.txt.gz) ---"
# 原始链接: https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

echo "--- 7. 正在解压 OWT-Sample 验证集... ---"
gunzip owt_valid.txt.gz

# --- 返回上级目录 ---
echo "--- 8. 所有文件已下载并解压完毕，返回上级目录 ---"
cd ..

echo "--- ✅ 数据集下载和准备任务完成 ---"