# SAM1 Evidence-Driven Refinement V1

基于用户提供的 SAM1 生成脚本重构；默认 **SAM1 ViT-H、逐框推理、风险样本触发、可靠负点启用、三池 CSV 字段向后兼容**。

## 文件说明

- `sam1_generate_refine.py`：运行入口、逐框合并、质量评估与文件导出。
- `sam_core.py`：SAM1 原生提示（`box` / `point_coords` / `mask_input`）。
- `evidence_refine.py`：风险诊断、局部高分辨率 RGB crop、跨视图可靠点、非 GT 安全门控、A4 选择性补边。
- `quality_metrics.py`：原始质量公式与异常诊断，尽量原样复用，不依赖 GT。
- `postprocess.py`：原先 3 种质量排名（用于与旧分池脚本兼容，不作为修复选择依据）。
- `evaluation.py`：只有此模块读取 GT，生成修复前后指标和面板可视化。
- `legacy_columns.py`：116 个旧字段按原始顺序保留，另附新增的修复结果字段。
- `configs/A0.json ... A4.json`：5 个消融配置。
- `tests/test_pipeline.py`：无需 GPU 和 SAM 权重的模拟集成测试。

## 安装

```bash
conda create -n sam1refine python=3.10 -y
conda activate sam1refine
# torch / torchvision 建议按你的 CUDA 版本从 pytorch.org 安装匹配版本。
pip install -r requirements.txt
```

如果 `segment-anything` 的镜像版本不正确，可使用官方实现：

```bash
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

在工程根目录执行。编辑 `config.py` 默认路径，或命令行覆盖以下四项。**默认包含 `--device cuda`；确保 CUDA 版 PyTorch 和显卡可用。**

```bash
python sam1_generate_refine.py \
  --config configs/A0.json \
  --checkpoint ./weights/sam_vit_h_4b8939.pth \
  --img-dir ./data/Train/Imgs \
  --box-dir ./data/Train/box_label \
  --gt-dir ./data/Train/GT \
  --output-dir ./SGT/SAM1_Refine_V1 \
  --max-images 10
```

A4 默认启动风险触发的完整精修：

```bash
python sam1_generate_refine.py --config configs/A4.json
```

全部五组串行（每个运行将自动输出到 `output_dir/A#`，不覆盖彼此）：

```bash
for a in 0 1 2 3 4; do
  python sam1_generate_refine.py --config configs/A${a}.json
done
```

无 GT 时：`python sam1_generate_refine.py --config configs/A4.json --no-gt-eval`。
为了单独观察算法对所有样本的作用，可附加 `--refine-scope all`；默认 `risk`。
关闭可靠负点：`--no-negative-points`。查看严格失败位置：`--fail-fast`。

## 消融定义（精修全部仅在触发风险时启动）

| 实验 | 局部 RGB crop | 可靠正/负点 | 原生 256×256 Logits | 局部选择性修改 |
|---|---|---|---|---|
| A0 | 否 | 否 | 否 | 否，保留 SAM1 baseline |
| A1 | 否 | 否 | 是 | 否，单纯 logits 回传对照 |
| A2 | 是 | 否 | 否 | 仅初始边界窄带 / 失效时局部替换 |
| A3 | 是 | 是 | 否 | 否，受几何安全门控 |
| A4 | 是 | 是 | 是 | 是，局部视图与二次结果共同支持新增/删除 |

- **Box** 始终是原始 LabelMe JSON 中的矩形框；多个框逐框推理，最终逻辑 OR 合并。`box_manifest.json` 的 `prompt_box` 是便于旧消费方读的 *envelope 记录*，**绝不作为 SAM 提示**；真正发送的框在 `prompt_boxes`。由于框重叠，最终前景允许发生交叠；没有最大连通域过滤。
- **局部 crop** 是从原始 RGB 重新 `set_image`，不是 crop 掩码；full-image SAM 的 low-res logits **绝不会直接作为 crop 的 mask_input**。在 A4 等原图第二轮解码时必须回到原始完整 RGB，并在相同原图特征坐标系传入 256×256 原生 logits。
- **可靠正点** 在初始 Mask 与局部视图 Mask 交集深处；**负点** 必须在原始 Box 外、局部 crop 可观测区域内，且两视图概率均低，无法验证时自动不添加。框内不产生负点。
- **多框负点安全**：负点禁止落在该图所有标注框的并集内，不会把另一个目标框内的像素当作当前框的背景点。
- **失效初始候选恢复**：当第一轮 Mask 近乎为空或极低分时，允许用通过质量门控的局部候选内部点尝试恢复；此时不传递可能错误的旧 Logits，并记录 `recovered_from_local_only=1`。
- **A4 选择性编辑**：补充像素须被局部视图和二次输出同时支持；删除原初 Mask 像素还要位于窄带且两者一致认为是背景。局部细长新增像素不被强行限制在原边缘窄带。
- **微小细结构编辑**：改用至少 3 个修改像素的要求，而不再默认要求改变整图 0.05% 的像素；新增像素还要求双视图概率证据达到配置阈值。
- 全局守卫（面积、Box 外扩张、SAM score、稳定性、对原 Mask 的 IoU）并不能证明质量提高，**只是启发式保护措施**。在初始 Mask 明显失效时放宽部分主体保持约束；测试重点是记录接受与回退的真实 GT 变化。
- **纯视觉 A1 没有额外独立证据，特意作为负对照**；真正的新线索来自局部高分辨率观测（仍是同一张 RGB 和同一个 SAM）。
- 仅 `evaluation.py` 使用 GT；无 GT 依然能够输出伪标签、质量 CSV 和三池需要的排名。

## 输出布局

```text
SGT/SAM1_Refine_V1/A4/
├── Initial_Masks/              # 初始逐框 OR 二值 Mask
├── Pred_Masks/                 # 最终逐框 OR 二值 Mask，用于训练
├── Prob_48x48_FP16/           # 最终预测的 48x48 fp16 软概率
├── LowRes_Logits/             # npz: logits [K,256,256] 原生初始 logits + boxes [K,4]
├── Compare_Vis/               # RGB/初始/最终/GT FP/FN 可视化
├── csv/
│   ├── image_level_metrics.csv  # 旧 116 字段及新增修复字段
│   └── refinement_analysis.csv  # 每张图每个框的触发/点/回退原因
├── box_manifest.json
├── evaluation_results.json
├── generation_config.json
└── errors.json
```

`refinement_analysis.csv` 是每框记录；`image_level_metrics.csv` 是合并后的每图记录。第一轮 SAM 的三个候选的整图共识在多框时定义为“每个候选序号先逐框 OR，再算三候选共识”；整图分数/稳定性使用最弱框（`min`）。因此**单框与原脚本一一对应，多框重新定义了这些聚合值**。

**旧质量字段语义**：`sam_score_selected`、`sam_stability_selected`、`candidate_consensus_s` 仍记录**初始 SAM 证据**，防止将融合掩码误标为 SAM 官方评分。`q_region` 与边界相关指标以最终融合 Mask 计算；`q_shape` 使用初始 SAM score、stability、consensus；`q_dual` 依旧几何平均。这是向后兼容的折中，后续分池时应把修复是否接受作为显式字段纳入分析。

`is_bad_response` 保留旧异常诊断定义，但 **`force_unvalue` 不再因为仅碎片多而强制拒绝**，以免多目标、遮挡遭到误杀。原先 high/middle/unvalue 标记只是“旧推荐池兼容信息”，不应被视作你最新 Clean/Noisy/Reject 三池最终分配结果。具体三池请继续在独立脚本按原阈值规则产生。

**旧兼容字段** `gt_bad_by_iou` / `gt_bad_by_mae` / `gt_is_bad` 继续使用原始脚本的 `IoU < 0.70` 或 `MAE > 0.05` 定义，由 `bad_iou_threshold`、`bad_mae_threshold` 配置。额外输出 `gt_bad_strict` (`IoU < 0.90` 或 `MAE > 0.02`) 和 `gt_bad_loose` (`IoU <0.80` 或 `MAE>0.03`) 供对照。**GT 不参与筛选与接受决策**。

如某张图处理失败，`errors.json` 写具体异常，并在总结中记录 `failed_count`；检查 `generated_count` 是否为训练集规模（预期 4040），不要直接拿缺失样本跑训练。

## 快速测试

```bash
python -m compileall -q .
pytest -q tests/test_pipeline.py
```

测试还覆盖多框负点排除、失效候选绕过错误 Logits 和小像素修改门控；所有模拟测试均可脱离权重执行。真实 SAM ViT-H 的输入与显存行为仍需在实际运行环境核验。

## 算法限制

- 这版故意**不引入** CRF、随机游走、轮廓演化、强平滑、强开运算和“只保留最大连通域”；它们会干扰对二次提示增益的解释。
- 二次提示没有 GT 质量提升的理论保证。A4 的门控阈值是初始启发式，尚未经你的 4040 张训练集和细结构子集验证；必须比较 A0–A4 的 MAE / IoU 及退化率。
- `score` 表示 SAM 的预测质量分数，不是真实 IoU。融合 Mask 的质量没有直接的 SAM 预测分数。
- 测试不需要 SAM 权重，**真实 SAM ViT-H 前向性能和实际收益尚需在你的 GPU 及数据集上验证**。
