import os
import cv2
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from contextlib import nullcontext

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================================================
# 1. 全局配置 (Config)
# =========================================================
class Config:
    SAM2_CHECKPOINT = "./sam_pseado/weights/sam2_hiera_large.pt"
    SAM2_CONFIG = "sam2_hiera_l.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DIR_IMG = "./data/Train/Imgs"
    DIR_SCRIBBLE = "./data/Train/Scribble"
    DIR_GT = "./data/Train/GT"
    DIR_JSON = "./data/Train/box_label"

    BASE_OUT_DIR = "./softresult/SAM2_Optimized3"
    DIR_BASE = os.path.join(BASE_OUT_DIR, "01_SoftLogits_Base")
    DIR_PROFILE = os.path.join(BASE_OUT_DIR, "02_SoftLogits_SO_MO")
    DIR_VIS = os.path.join(BASE_OUT_DIR, "visualizations")
    CSV_PATH = os.path.join(BASE_OUT_DIR, "soft_so_mo_metrics.csv")
    JSON_SUMMARY_PATH = os.path.join(BASE_OUT_DIR, "summary_metrics.json")

    TARGET_SIZE = (1024, 1024)
    SEED = 0

    SO_THR = 0.10
    REFINE_ITERS = 2
    BOX_PROMPT_MARGIN = 5
    MO_MERGE_BOX_MARGIN = 25

    VIS_MAE_TH = 0.030
    VIS_GAIN_TH = 0.003


# =========================================================
# 2. 严谨的学术评估指标 (COD Metrics)
# =========================================================
class MetricsEvaluator:
    @staticmethod
    def prepare_pred_gt(pred, gt):
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = pred.astype(np.float32)
        if pred.max() > 1.0: pred /= 255.0
        pred = np.clip(pred, 0.0, 1.0)
        gt = (gt > 128).astype(np.float32)
        return pred, gt

    @staticmethod
    def compute_mae(pred, gt):
        return float(np.mean(np.abs(pred - gt)))

    @staticmethod
    def object_score(pred_region):
        if pred_region.size == 0: return 0.0
        x, sigma_x = pred_region.mean(), pred_region.std()
        return float((2.0 * x) / (x * x + 1.0 + sigma_x + 1e-8))

    @staticmethod
    def s_object(pred, gt):
        fg, bg = pred[gt == 1], 1.0 - pred[gt == 0]
        u = gt.mean()
        return float(u * MetricsEvaluator.object_score(fg) + (1.0 - u) * MetricsEvaluator.object_score(bg))

    @staticmethod
    def centroid(gt):
        h, w = gt.shape
        if gt.sum() == 0: return w // 2, h // 2
        ys, xs = np.where(gt > 0.5)
        return max(1, min(w - 1, int(np.round(xs.mean())))), max(1, min(h - 1, int(np.round(ys.mean()))))

    @staticmethod
    def ssim_region(pred, gt):
        if pred.size == 0: return 0.0
        x, y = pred.mean(), gt.mean()
        sx, sy = ((pred - x) ** 2).mean(), ((gt - y) ** 2).mean()
        sxy = ((pred - x) * (gt - y)).mean()
        return float((4.0 * x * y * sxy) / ((x * x + y * y) * (sx + sy) + 1e-8))

    @staticmethod
    def s_region(pred, gt):
        h, w = gt.shape
        x, y = MetricsEvaluator.centroid(gt)
        area = float(h * w)
        return float(
            (x * y / area) * MetricsEvaluator.ssim_region(pred[:y, :x], gt[:y, :x])
            + ((w - x) * y / area) * MetricsEvaluator.ssim_region(pred[:y, x:], gt[:y, x:])
            + (x * (h - y) / area) * MetricsEvaluator.ssim_region(pred[y:, :x], gt[y:, :x])
            + ((w - x) * (h - y) / area) * MetricsEvaluator.ssim_region(pred[y:, x:], gt[y:, x:])
        )

    @classmethod
    def compute_smeasure(cls, pred, gt, alpha=0.5):
        y = gt.mean()
        if y == 0: return float(1.0 - pred.mean())
        if y == 1: return float(pred.mean())
        return float(max(0.0, min(1.0, alpha * cls.s_object(pred, gt) + (1.0 - alpha) * cls.s_region(pred, gt))))

    @staticmethod
    def compute_f_binary(pred, gt, beta2=0.3):
        gt_sum = gt.sum()
        if gt_sum == 0: return 0.0
        bin_pred = (pred >= 0.5).astype(np.float32)
        tp = (bin_pred * gt).sum()
        precision = tp / (bin_pred.sum() + 1e-8)
        recall = tp / (gt_sum + 1e-8)
        return float((1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8))

    @staticmethod
    def compute_e_binary(pred, gt):
        fm, gt = (pred >= 0.5).astype(np.float32), gt.astype(np.float32)
        if gt.sum() == 0: return float(1.0 - fm.mean())
        if gt.sum() == gt.size: return float(fm.mean())
        fm_align, gt_align = fm - fm.mean(), gt - gt.mean()
        align = 2.0 * gt_align * fm_align / (gt_align * gt_align + fm_align * fm_align + 1e-8)
        return float((((align + 1.0) ** 2) / 4.0).mean())

    @classmethod
    def evaluate(cls, pred_raw, gt_raw):
        pred, gt = cls.prepare_pred_gt(pred_raw, gt_raw)
        return {
            "Smeasure": cls.compute_smeasure(pred, gt),
            "maxE": cls.compute_e_binary(pred, gt),
            "maxF": cls.compute_f_binary(pred, gt),
            "MAE": cls.compute_mae(pred, gt),
        }


# =========================================================
# 3. 数据流控制 (DataLoader)
# =========================================================
class DataLoader:
    # (同上一版的纯粹工具函数，略去冗余注释，全保留)
    @staticmethod
    def list_images(folder):
        paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(glob(os.path.join(folder, ext)))
            paths.extend(glob(os.path.join(folder, ext.upper())))
        return sorted(set(paths))

    @staticmethod
    def find_file(folder, base_name, exts):
        for ext in exts:
            for case_ext in [ext, ext.upper()]:
                p = os.path.join(folder, base_name + case_ext)
                if os.path.exists(p): return p
        return None

    @staticmethod
    def load_boxes(json_path, w, h):
        if not json_path or not os.path.exists(json_path): return []
        boxes = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                pts = shape.get("points", [])
                if len(pts) < 2: continue
                xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
                x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(w - 1, max(xs)), min(h - 1, max(ys))
                if x2 > x1 and y2 > y1: boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
        return boxes

    @staticmethod
    def get_points_from_scribble(scribble_mask):
        points, labels = [], []
        for class_val, sam_label in [(1, 1), (2, 0)]:
            class_mask = (scribble_mask == class_val).astype(np.uint8)
            if class_val == 1 and class_mask.sum() == 0 and scribble_mask.max() > 2:
                class_mask = (scribble_mask > 128).astype(np.uint8)
            y_idx, x_idx = np.where(class_mask > 0)
            if len(y_idx) == 0: continue
            dist_map = cv2.distanceTransform(class_mask, cv2.DIST_L2, 5)
            ymin, ymax = y_idx.min(), y_idx.max()
            xmin, xmax = x_idx.min(), x_idx.max()
            y_steps, x_steps = np.unique(np.linspace(ymin, ymax, 4, dtype=int)), np.unique(
                np.linspace(xmin, xmax, 4, dtype=int))
            for i in range(len(y_steps) - 1):
                for j in range(len(x_steps) - 1):
                    y0, y1, x0, x1 = y_steps[i], y_steps[i + 1], x_steps[j], x_steps[j + 1]
                    m = (y_idx >= y0) & (y_idx <= y1) & (x_idx >= x0) & (x_idx <= x1)
                    vy, vx = y_idx[m], x_idx[m]
                    if len(vy) == 0: continue
                    k = int(np.argmax(dist_map[vy, vx]))
                    points.append([vx[k], vy[k]])
                    labels.append(sam_label)
        if not points: return None, None
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


# =========================================================
# 4. 纯 GPU 推理引擎 + 记忆主动采样 (SAM2Engine)
# =========================================================


# =========================================================
# 4. 纯 GPU 推理引擎 + 记忆主动采样 (SAM2Engine)
# =========================================================
class SAM2Engine:
    def __init__(self, config):
        self.device = config.DEVICE
        self.model = build_sam2(config.SAM2_CONFIG, config.SAM2_CHECKPOINT, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.iters = config.REFINE_ITERS
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True

    def set_image(self, image_rgb):
        self.predictor.set_image(image_rgb)

    def _sample_uncertainty_point(self, logits_raw, scribble_mask, original_shape):
        if isinstance(logits_raw, np.ndarray):
            t_logits = torch.from_numpy(logits_raw)
        else:
            t_logits = logits_raw.clone()

        t_logits = t_logits.squeeze().unsqueeze(0).unsqueeze(0).to(torch.float32)
        logits_full = F.interpolate(t_logits, size=original_shape, mode='bilinear',
                                    align_corners=False).squeeze().cpu().numpy()

        # 数值安全的 Sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(logits_full, -10.0, 10.0)))
        uncertainty = 1.0 - np.abs(2.0 * p - 1.0)

        valid_mask = (scribble_mask == 1) | (scribble_mask == 2)
        if scribble_mask.max() > 2: valid_mask = (scribble_mask > 0)
        if not np.any(valid_mask): return None, None

        masked_uncertainty = uncertainty * valid_mask.astype(np.float32)
        y, x = np.unravel_index(np.argmax(masked_uncertainty), masked_uncertainty.shape)

        val = scribble_mask[y, x]
        label = 1 if val == 1 or val > 128 else 0
        return np.array([[x, y]], dtype=np.float32), np.array([label], dtype=np.int32)

    def refine_predict(self, initial_points, initial_labels, box, scribble_mask=None, original_shape=None,
                       iters_override=None):
        current_points, current_labels = initial_points, initial_labels
        memory_logits, prob_out = None, None

        # 允许外部控制迭代次数（用于消融实验 Base 阻断）
        total_iters = self.iters if iters_override is None else iters_override

        with torch.inference_mode(), torch.autocast(device_type=self.device,
                                                    dtype=torch.bfloat16) if self.device == "cuda" else nullcontext():
            for step in range(total_iters + 1):
                if memory_logits is not None and memory_logits.ndim == 3:
                    memory_logits = np.expand_dims(memory_logits, axis=0)

                # 开启 return_logits=True，此时 masks 返回的实际上是 Logits 数组
                masks, scores, logits = self.predictor.predict(
                    point_coords=current_points, point_labels=current_labels,
                    box=box[None, :] if box is not None else None,
                    mask_input=memory_logits, multimask_output=False, return_logits=True
                )

                memory_logits = logits

                # 【核心修复】: 将 Logits 严格通过 Sigmoid 转换为 [0, 1] 概率图！绝不能除以 255！
                raw_logits = masks[0].astype(np.float32)
                prob_out = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -10.0, 10.0)))

                # 触发主动采样纠错
                if step < total_iters and scribble_mask is not None and original_shape is not None:
                    new_pt, new_lbl = self._sample_uncertainty_point(logits, scribble_mask, original_shape)
                    if new_pt is not None:
                        if current_points is None:
                            current_points, current_labels = new_pt, new_lbl
                        else:
                            current_points = np.vstack((current_points, new_pt))
                            current_labels = np.append(current_labels, new_lbl)
        return prob_out



# =========================================================
# 5. 流水线调度 (Pipeline)
# =========================================================
class Pipeline:
    def __init__(self, config):
        self.cfg = config
        self.engine = SAM2Engine(config)
        self.metrics_list = []
        for d in [self.cfg.BASE_OUT_DIR, self.cfg.DIR_BASE, self.cfg.DIR_PROFILE, self.cfg.DIR_VIS]:
            os.makedirs(d, exist_ok=True)

    def run_normal(self, img_1024, points, labels, box_1024, scribble, orig_shape, iters_override=None):
        self.engine.set_image(img_1024)
        return self.engine.refine_predict(points, labels, box_1024, scribble, orig_shape, iters_override)

    def run_mo(self, image_bgr, scribble, boxes):
        h, w = image_bgr.shape[:2]
        image_1024 = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), self.cfg.TARGET_SIZE)
        scribble_1024 = cv2.resize(scribble, self.cfg.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        self.engine.set_image(image_1024)
        merged_1024 = np.zeros(self.cfg.TARGET_SIZE[::-1], dtype=np.float32)

        for box in boxes:
            sx, sy = self.cfg.TARGET_SIZE[0] / w, self.cfg.TARGET_SIZE[1] / h
            b1024 = np.clip([box[0] * sx - self.cfg.BOX_PROMPT_MARGIN, box[1] * sy - self.cfg.BOX_PROMPT_MARGIN,
                             box[2] * sx + self.cfg.BOX_PROMPT_MARGIN, box[3] * sy + self.cfg.BOX_PROMPT_MARGIN], 0,
                            1023).astype(np.float32)

            x1, y1, x2, y2 = map(int, np.round(b1024))
            local_sc_1024 = np.zeros_like(scribble_1024)
            local_sc_1024[y1:y2 + 1, x1:x2 + 1] = scribble_1024[y1:y2 + 1, x1:x2 + 1]
            pts, lbls = DataLoader.get_points_from_scribble(local_sc_1024)

            # MO 分支默认享受迭代和纠错红利
            prob_box = self.engine.refine_predict(pts, lbls, b1024, local_sc_1024, self.cfg.TARGET_SIZE[::-1])

            gate = np.zeros_like(prob_box)
            gx1, gy1, gx2, gy2 = max(0, x1 - self.cfg.MO_MERGE_BOX_MARGIN), max(0,
                                                                                y1 - self.cfg.MO_MERGE_BOX_MARGIN), min(
                1023, x2 + self.cfg.MO_MERGE_BOX_MARGIN), min(1023, y2 + self.cfg.MO_MERGE_BOX_MARGIN)
            gate[gy1:gy2 + 1, gx1:gx2 + 1] = 1.0
            merged_1024 = np.maximum(merged_1024, prob_box * gate)

        return cv2.resize(merged_1024, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_image(self, img_path):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        sc_path = DataLoader.find_file(self.cfg.DIR_SCRIBBLE, base_name, [".png", ".jpg"])
        gt_path = DataLoader.find_file(self.cfg.DIR_GT, base_name, [".png", ".jpg"])
        js_path = DataLoader.find_file(self.cfg.DIR_JSON, base_name, [".json"])

        if not sc_path or not gt_path: return None

        img_bgr = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        scribble = cv2.imread(sc_path, cv2.IMREAD_GRAYSCALE)
        if scribble.shape != img_bgr.shape[:2]: scribble = cv2.resize(scribble, (img_bgr.shape[1], img_bgr.shape[0]),
                                                                      interpolation=cv2.INTER_NEAREST)

        h, w = img_bgr.shape[:2]
        boxes = DataLoader.load_boxes(js_path, w, h)
        num_boxes = len(boxes)
        union_b = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                   max(b[3] for b in boxes)] if boxes else None
        area_ratio = (union_b[2] - union_b[0]) * (union_b[3] - union_b[1]) / (h * w) if union_b else 0.0

        img_1024 = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), self.cfg.TARGET_SIZE)
        sc_1024 = cv2.resize(scribble, self.cfg.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        pts, lbls = DataLoader.get_points_from_scribble(sc_1024)
        b1024 = np.array(
            [union_b[0] * (1024 / w), union_b[1] * (1024 / h), union_b[2] * (1024 / w), union_b[3] * (1024 / h)],
            dtype=np.float32) if union_b else None

        if pts is None and b1024 is None: return None

        # ========================================================
        # 严格消融对比设计
        # ========================================================
        # [Base - Old]: 强制迭代 0 次 (单次前向)，关闭 Active 采样
        base_prob = self.run_normal(img_1024, pts, lbls, b1024, scribble=None, orig_shape=None, iters_override=0)
        base_prob = cv2.resize(base_prob, (w, h), interpolation=cv2.INTER_LINEAR)

        # [Ours - New]: 动态路由 + 主动纠错采样 (默认享受 2 轮迭代红利)
        if num_boxes > 1:
            route = "MO_PER_BOX"
            prof_prob = self.run_mo(img_bgr, scribble, boxes)
        elif num_boxes == 1 and 0 < area_ratio <= self.cfg.SO_THR:
            route = "SO_CROP"
            prof_prob = self.run_normal(img_1024, pts, lbls, b1024, scribble, (h, w))
            prof_prob = cv2.resize(prof_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            route = "NORMAL"
            prof_prob = self.run_normal(img_1024, pts, lbls, b1024, scribble, (h, w))
            prof_prob = cv2.resize(prof_prob, (w, h), interpolation=cv2.INTER_LINEAR)

        # 二值化输出保存
        base_mask, prof_mask = (base_prob > 0.5).astype(np.uint8) * 255, (prof_prob > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(self.cfg.DIR_BASE, f"{base_name}.png"), base_mask)
        cv2.imwrite(os.path.join(self.cfg.DIR_PROFILE, f"{base_name}.png"), prof_mask)

        # ====== 严谨预测评估计算 ======
        b_m = MetricsEvaluator.evaluate(base_mask, gt)
        p_m = MetricsEvaluator.evaluate(prof_mask, gt)

        # 可视化输出
        gain_mae = b_m["MAE"] - p_m["MAE"]
        if route != "NORMAL" or b_m["MAE"] > self.cfg.VIS_MAE_TH or abs(gain_mae) > self.cfg.VIS_GAIN_TH:
            self._make_vis(img_bgr, gt, base_mask, prof_mask, b_m, p_m, route,
                           f"boxes={num_boxes}, area={area_ratio:.3f}", base_name)

        return {
            "image_name": img_name, "base_name": base_name, "route_type": route,
            "num_boxes": num_boxes, "box_area_ratio": area_ratio, "has_box": int(union_b is not None),
            "base_Smeasure": b_m["Smeasure"], "base_maxE": b_m["maxE"], "base_maxF": b_m["maxF"],
            "base_MAE": b_m["MAE"],
            "profile_Smeasure": p_m["Smeasure"], "profile_maxE": p_m["maxE"], "profile_maxF": p_m["maxF"],
            "profile_MAE": p_m["MAE"],
            "delta_S_new_minus_base": p_m["Smeasure"] - b_m["Smeasure"],
            "delta_E_new_minus_base": p_m["maxE"] - b_m["maxE"],
            "delta_F_new_minus_base": p_m["maxF"] - b_m["maxF"],
            "delta_MAE_base_minus_new": b_m["MAE"] - p_m["MAE"],
        }

    def _make_vis(self, img, gt, b_mask, p_mask, b_m, p_m, route, info, name):
        def panel(x): return cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (256, 256))

        v_img = cv2.resize(img, (256, 256))
        v_gt, v_base, v_prof = panel(gt), panel(b_mask), panel(p_mask)
        cv2.putText(v_img, route, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(v_base, f"Base MAE:{b_m['MAE']:.4f}", (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(v_prof, f"New MAE:{p_m['MAE']:.4f}", (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(self.cfg.DIR_VIS, f"{name}_abl.png"),
                    np.concatenate([v_img, v_gt, v_base, v_prof], axis=1))


# =========================================================
# 6. 主程序入口 (Main)
# =========================================================
def summarize(rows, prefix):
    out = {}
    for m in ["Smeasure", "maxE", "maxF", "MAE"]:
        k = f"{prefix}_{m}"
        vals = [r[k] for r in rows if isinstance(r.get(k), float) and np.isfinite(r[k])]
        out[m] = float(np.mean(vals)) if vals else 0.0
    return out


def summarize_route(rows, route_type, prefix):
    sub = [r for r in rows if r["route_type"] == route_type]
    out = summarize(sub, prefix)
    out["num_images"] = len(sub)
    return out


def main():
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    print("=" * 70)
    print(" SAM2 Active Memory Pipeline - Evaluation & Ablation Mode ")
    print("=" * 70)

    cfg = Config()
    pipe = Pipeline(cfg)
    img_paths = DataLoader.list_images(cfg.DIR_IMG)

    rows = []
    for p in tqdm(img_paths, desc="Evaluating COD Metrics"):
        res = pipe.process_image(p)
        if res: rows.append(res)

    if not rows:
        print("No valid images processed.")
        return

    # 保存 CSV
    with open(cfg.CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 聚合 JSON
    summary = {
        "SoftLogits_Base": summarize(rows, "base"),
        "ActiveMemory_New": summarize(rows, "profile"),
        "RouteBreakdown": {
            "NORMAL": summarize_route(rows, "NORMAL", "profile"),
            "SO_CROP": summarize_route(rows, "SO_CROP", "profile"),
            "MO_PER_BOX": summarize_route(rows, "MO_PER_BOX", "profile"),
        },
        "Diagnostics": {
            "num_images": len(rows),
            "mean_delta_S": float(np.mean([r["delta_S_new_minus_base"] for r in rows])),
            "mean_delta_MAE": float(np.mean([r["delta_MAE_base_minus_new"] for r in rows])),
        }
    }

    with open(cfg.JSON_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 25 + " Ablation Results " + "=" * 25)
    b = summary["SoftLogits_Base"]
    p = summary["ActiveMemory_New"]
    print(f"[Base - Old] | S={b['Smeasure']:.4f} | E={b['maxE']:.4f} | F={b['maxF']:.4f} | MAE={b['MAE']:.4f}")
    print(f"[Ours - New] | S={p['Smeasure']:.4f} | E={p['maxE']:.4f} | F={p['maxF']:.4f} | MAE={p['MAE']:.4f}")
    print(
        f"-> Improvement: dS = {summary['Diagnostics']['mean_delta_S']:+.4f}, dMAE = {summary['Diagnostics']['mean_delta_MAE']:+.4f}")
    print("=" * 68)
    print(f"Metrics saved to: {cfg.CSV_PATH}")


if __name__ == "__main__":
    main()