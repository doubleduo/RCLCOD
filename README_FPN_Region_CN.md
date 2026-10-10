# RCLCOD FPN 最小可靠区域实验

本补丁基于 `doubleduo/RCLCOD` commit `af1ae497aa87c4d03c0d674a5c59a0de0e5b0069` 的 `PvtV2B4_FPN_Baseline`，不修改骨干和推理图。

## 文件

- `fpn_region_ablation.py`: 新的 `PvtV2B4_FPN_RegionAblation` 模型（继承 FPN，仅替换损失）。
- `install_fpn_region_ablation.py`: 拷贝文件、注册模型，创建三种配置，可重复执行。
- `run_fpn_region_ablation.sh`: 运行三组实验。

## 安装、运行

将前三个文件复制到 RCLCOD 根目录，执行：

```bash
python install_fpn_region_ablation.py --epochs 40
bash run_fpn_region_ablation.sh
```

只运行某一组：

```bash
MODES="seed" bash run_fpn_region_ablation.sh
```

改为完整 150 epoch：

```bash
python install_fpn_region_ablation.py --epochs 150
bash run_fpn_region_ablation.sh
```

可选择新的三池名单：

```bash
python install_fpn_region_ablation.py --epochs 40 \
  --high-list ./data/pseudo_pool/sam2refine/high.txt \
  --middle-list ./data/pseudo_pool/sam2refine/middle.txt
```

所有配置固定 High/Middle 的池级抽样比例 1:1，且使用同一模型初始化/训练轮次。每个 Epoch 4040 次有放回抽样，维持现有随机种子、Batch Size、学习率及评价间隔。

## 三组实验

| 实验 | High | Middle | UAL |
|---|---|---|---|
| baseline | 完整 BCE | 完整 BCE | 无 |
| seed | 完整 BCE | Box 约束的可靠前景/背景 BCE | 无 |
| adaptive | 完整 BCE | seed + 基于当前 Student 置信和伪标签一致性的边界 BCE（默认系数 0.25） | 无 |

备注：**FPN 原始 `PvtV2B4_FPN_Baseline` 只使用 BCE，本来就没有 UAL**。本实现不向 Middle 添加 UAL，也不对 High 额外添加 UAL。

## 可靠区域定义

- 前景：SAM2 精修二值 mask 的腐蚀内核（5x5），并且位于 Box 内；如果细小前景腐蚀后为空，保留原前景与 Box 的交集。
- 背景：扩大 Box（7x7）外的像素。
- 未知区域：不直接监督；adaptive 仅对当前预测较确定且与 SAM2 一致的 *Box 内掩码边界带* 赋予柔性权重。
- 未引入 Teacher、时序可信度或额外图像编码。`adaptive` 是低成本的初步验证，不应宣传为完整动态课程学习。

## 调试和注意事项

1. `configs/sam2_real.py` 仍是配置基线，安装程序只新建 `configs/rcl_fpn_region/fpn_*.py`。
2. 设置 `train.data.use_box=True`，需要 `dataset.yaml` 对 `combined_tr` 配有 `box_json`，与原来的 Box 标注位置一致。
3. 默认分池名单沿用 `configs/sam2_real.py` 指向的 `high.txt` 与 `middle.txt`；如果验证 Rank V2，三组实验都要使用新名单，不能混用。
4. 检查 `loss_metrics.csv` 里的 `middle_seed`、`middle_dynamic`、`fg_coverage`、`bg_coverage`、`band_gate`。只有 `adaptive` 的 `middle_dynamic` 应非零。
5. 原 FPN BCE 为全图平均；新的 seed BCE 将前景与背景分别归一化后等权平均。应由单独消融判定改变权重与空间监督的效应，不能仅凭最终指标归因于动态性。
6. 此脚本已在 CPU 模拟网络上做前向/反向测试，尚未使用用户 GPU 与真实数据执行完整训练。
