#!/bin/bash

export PYTHONPATH=/home/ericxhzou/Code/PCGL-Benchmark

DEBUG=0
TRAIN_AUG=0  # for training of place recognition (do not use data augmentation, LoGG3DNet has bugs)
EVAL_AUG=2  # for evaluation of place recognition
TRAIN_ENV=Hankou_1_2
PICKLE_ROOT=/home/ericxhzou/Code/PCGL-Benchmark/pickles
SUBMAP_TYPE=30m_2m_no_dynamics
CONFIG_ROOT=/home/ericxhzou/Code/PCGL-Benchmark/config
MODEL_ROOT=/home/ericxhzou/Code/PCGL-Benchmark/exp/pr
LOG_FILE="/home/ericxhzou/Code/PCGL-Benchmark/exp/log/$(date).txt"


{
echo "=== 脚本开始执行: $(date) ==="

# 函数：执行命令并检查状态
execute_command() {
    local cmd="$1"
    local description="$2"
    
    echo "========================================"
    echo "开始执行: $description"
    echo "命令: $cmd"
    echo "----------------------------------------"
    
    # 执行命令
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ 完成: $description"
        echo "========================================"
        return 0
    else
        echo "✗ 失败: $description (退出码: $exit_code)"
        echo "========================================"
        return $exit_code
    fi
}


## preprocess datasets
execute_command "python data_prep/Wuhan/pre_process.py --root /home/ericxhzou/Data/WHU-PCGL/PublishData-V2/Merge --save_dir $PICKLE_ROOT/Wuhan --setting $SUBMAP_TYPE" "武汉数据预处理"
execute_command "python data_prep/Wuhan/generate_train.py --root $PICKLE_ROOT/Wuhan --save_dir $PICKLE_ROOT/Wuhan --setting $SUBMAP_TYPE" "生成武汉训练数据"
execute_command "python data_prep/Wuhan/generate_test.py --root $PICKLE_ROOT/Wuhan --save_dir $PICKLE_ROOT/Wuhan --setting $SUBMAP_TYPE" "生成武汉测试数据"

execute_command "python data_prep/Oxford/generate_train.py --root /home/ericxhzou/Data/benchmark_datasets/Public_Dataset/Oxford --save_dir $PICKLE_ROOT/Oxford" "生成牛津训练数据"
execute_command "python data_prep/Oxford/generate_test.py --root /home/ericxhzou/Data/benchmark_datasets/Public_Dataset/Oxford --save_dir $PICKLE_ROOT/Oxford" "生成牛津测试数据"


## place recognition
PR_METHODS=("PointNetVLAD" "PPTNet" "MinkLoc3D" "EgoNN" "LoGG3DNet")
echo "开始位置识别..."
for pr_method in "${PR_METHODS[@]}"; do
    execute_command "python train/train_pr.py --config $CONFIG_ROOT/pr/${pr_method}.yaml --train_aug $TRAIN_AUG --debug $DEBUG" "训练 ${pr_method}"
done


## rerank for place recognition
RERANK_METHODS=("AlphaQE" "AverageQE" "SpectralGV" "RankPointRetrieval" "RANSAC")
PR_BACKBONES=("PPTNet" "EgoNN" "LoGG3DNet")
echo "开始重排序..."
for rerank_method in "${RERANK_METHODS[@]}"; do
    for pr_backbone in "${PR_BACKBONES[@]}"; do
        execute_command "python eval/evaluate_rerank.py --config $CONFIG_ROOT/rerank/${rerank_method}.yaml --pr_backbone $pr_backbone --debug $DEBUG" "重排序 ${rerank_method} + ${pr_backbone}"
    done
done


## robustness to viewpoint change
echo "开始评估视角变化鲁棒性..."
for pr_method in "${PR_METHODS[@]}"; do
    for theta in {30..180..30}; do
        execute_command "python eval/evaluate_pr.py --config $CONFIG_ROOT/pr/${pr_method}.yaml --train_aug $TRAIN_AUG --eval_aug $EVAL_AUG --save_desc 0 --rot_theta $theta --debug $DEBUG" "视角鲁棒性 ${pr_method} θ=$theta"
    done
done

## analyse rerank result: LoGG3DNet+SGV
echo "开始分析LoGG3DNet+SGV的位置识别结果..."
execute_command "python datasets/rerank_dataset.py"

echo "所有任务执行完成!"
echo "=== 脚本执行结束: $(date) ==="
} | tee "$LOG_FILE"
