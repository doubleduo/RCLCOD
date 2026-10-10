"""Non-GT postprocessing: preserve original ranking semantics for downstream pools."""
from quality_metrics import safe_float
import numpy as np

def assign_q_ranks(image_rows):
    """
    在 high/middle 可用样本中分别保存：
      - 区域可靠性排名 q_region；
      - 形状可靠性排名 q_shape；
      - 双维平衡参考排名 q_dual。

    q_rank_global / q_rank_within_pool 保留为 q_dual 的兼容别名。
    """
    metric_specs = [
        ("q_region", "q_region_rank_global", "q_region_rank_within_pool"),
        ("q_shape", "q_shape_rank_global", "q_shape_rank_within_pool"),
        ("q_dual", "q_dual_rank_global", "q_dual_rank_within_pool"),
    ]

    eligible_indices = [
        idx
        for idx, row in enumerate(image_rows)
        if row["recommended_pool"] in {"high", "middle"}
    ]

    for metric_name, global_field, pool_field in metric_specs:
        sorted_global = sorted(
            eligible_indices,
            key=lambda idx: safe_float(
                image_rows[idx].get(metric_name),
                default=-1.0,
            ),
            reverse=True,
        )
        for rank, idx in enumerate(sorted_global, start=1):
            image_rows[idx][global_field] = rank

        for pool_name in ["high", "middle"]:
            pool_indices = [
                idx
                for idx in eligible_indices
                if image_rows[idx]["recommended_pool"] == pool_name
            ]
            pool_indices.sort(
                key=lambda idx: safe_float(
                    image_rows[idx].get(metric_name),
                    default=-1.0,
                ),
                reverse=True,
            )
            for rank, idx in enumerate(pool_indices, start=1):
                image_rows[idx][pool_field] = rank

    for row in image_rows:
        row["q_rank_global"] = row.get("q_dual_rank_global", -1)
        row["q_rank_within_pool"] = row.get("q_dual_rank_within_pool", -1)
