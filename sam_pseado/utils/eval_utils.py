import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
from PIL import Image
from torchvision.utils import save_image
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    # elif cfg.prompt == "point":
    else:
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    # else:
    #     raise ValueError("Prompt Type Error!")
    return prompts

class Logits_Saver:
    def __init__(self, root):
        self.root = root
        self.dict = {}
    def add(self, imgae_name, embed):
        name, _ = os.path.splitext(imgae_name)
        self.dict[name] = embed.cpu()

    def save(self):
        if self.get_size() > 0:
            torch.save(self.dict, self.root+'/logits.pt')
    def get_size(self):
        return len(self.dict)

def save_logits_as_npy(logits, file_path="logits.npy"):
    name, _ = os.path.splitext(file_path)
    file_path = name + '.npy'
    
    logits_cpu = logits.cpu()

    logits_np = logits_cpu.numpy()

    np.save(file_path, logits_np)

def save_mask_as_image(mask, file_path="mask.png", threshold=0.5):
    mask_cpu = mask.cpu()

    mask_bin = mask_cpu > threshold

    mask_uint8 = mask_bin.to(torch.uint8) * 255

    mask_img = mask_uint8.squeeze(0)

    mask_np = mask_img.numpy()
    img = Image.fromarray(mask_np)

    img.save(file_path)
    # print(f"Mask image saved to {file_path}")

data_dict = {
    "cod" : "COD10K",
    "camo" : "CAMO",
    "cham" : "CHAMELEON"
}
def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0, save_image_path: str = None, is_val: bool = True, eval_dataset: str = None):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    cnt = 0
    eval_dataset = eval_dataset if eval_dataset else cfg['eval_dataset']
    if save_image_path:
        save_image_path = save_image_path + cfg['dataset'] + f"/{cfg['prompt']}/val/{data_dict[eval_dataset]}/" if is_val else save_image_path + cfg['dataset'] + f"/{cfg['prompt']}/train/"
    
    logits_saver = Logits_Saver(save_image_path)
    
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            if is_val:
                images, bboxes, gt_masks, images_name = data 
                images_list = list(torch.unbind(images, dim=0))
                num_images = images.size(0)
                prompts = get_prompts(cfg, bboxes, gt_masks)
            else:
                images, bboxes, gt_masks, key_points, kp_labels, images_name = data 
                images_list = list(torch.unbind(images, dim=0))
                num_images = images.size(0)
                
                if cfg.prompt == "box" or cfg.prompt == "coarse":
                    prompts = bboxes
                else:
                    prompts = [(x.unsqueeze(0), y.unsqueeze(0)) for x, y in zip(key_points, kp_labels)]
                
                # prompts = get_prompts(cfg, bboxes, gt_masks)

            image_embeddings, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask, image, image_name in zip(pred_masks, gt_masks, images_list, images_name):
                if save_image_path: 
                    device_id = pred_mask.get_device()
                    if not os.path.exists(f'{save_image_path}/pred'):
                        os.makedirs(f'{save_image_path}/pred')
                    if not os.path.exists(f'{save_image_path}/gt'):
                        os.makedirs(f'{save_image_path}/gt')
                    if not os.path.exists(f'{save_image_path}/image'):
                        os.makedirs(f'{save_image_path}/image')
                    if not os.path.exists(f'{save_image_path}/logit'):
                        os.makedirs(f'{save_image_path}/logit')
                    if not is_val:
                        # logits_saver.add(image_name, image_embeddings)
                        save_logits_as_npy(pred_mask, file_path=os.path.join(save_image_path, f"logit/{image_name}"))
                    save_image(image, os.path.join(save_image_path, f"image/{image_name}"))
                    save_mask_as_image(pred_mask, file_path=os.path.join(save_image_path, f"pred/{image_name}"))
                    save_mask_as_image(gt_mask, file_path=os.path.join(save_image_path, f"gt/{image_name}"))
                    cnt += 1
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    logits_saver.save()
    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def unspervised_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    init_prompt = cfg.prompt
    cfg.prompt = "box"
    iou_box, f1_box = validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = "point"
    iou_point, f1_point = validate(fabric, cfg, model, val_dataloader, name, iters)
    # cfg.prompt = "coarse"
    # validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = init_prompt
    return iou_box, f1_box, iou_point, f1_point


def contrast_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0, loss: float = 0.):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters, "loss": loss}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"metrics-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg
