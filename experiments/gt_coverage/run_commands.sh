#!/usr/bin/env bash
set -e

# full_gt
python basemain_continuous.py --config /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/configs/full_gt_n4040_seed112358.py --data-cfg /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/dataset_gt.yaml --model-name PvtV2B4_FPN_Baseline --output-dir outputs/gt_coverage/full_gt_n4040_seed112358

# random_gt
python basemain_continuous.py --config /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/configs/random_gt_n3365_seed112358.py --data-cfg /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/dataset_gt.yaml --model-name PvtV2B4_FPN_Baseline --output-dir outputs/gt_coverage/random_gt_n3365_seed112358

# random_pseudo
python basemain_continuous.py --config /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/configs/random_pseudo_n3365_seed112358.py --data-cfg /media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/experiments/gt_coverage/dataset_pseudo.yaml --model-name PvtV2B4_FPN_Baseline --output-dir outputs/gt_coverage/random_pseudo_n3365_seed112358
