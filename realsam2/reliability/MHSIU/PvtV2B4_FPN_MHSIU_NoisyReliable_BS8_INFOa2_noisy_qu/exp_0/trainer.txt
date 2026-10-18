"""Select plain FPN or FPN-MHSIU reliability model from the Bash command."""

import argparse
from functools import partial
from pathlib import Path
import shutil
import sys

import basemain_continuous as continuous
from methods.fpn_noisy_reliability import PvtV2B4_FPN_NoisyReliable
from methods.fpn_mhsiu_noisy_reliability import (
    MODES,
    PvtV2B4_FPN_MHSIU_NoisyReliable,
)
from utils.data_noisy_reliability import NoisyReliabilityDataset


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = (
    PROJECT_ROOT / "configs/curablation/fpn_noisy_reliable.py"
)
DATA_CONFIG_PATH = PROJECT_ROOT / "dataset.yaml"
PROBABILITY_DIR = (
    PROJECT_ROOT / "SGT/SAM2_RefineReliable_Box/Prob_48x48_FP16"
)
OUTPUT_DIR = "realsam2/noisy_reliability"

MODEL_CLASSES = {
    "PvtV2B4_FPN_NoisyReliable": PvtV2B4_FPN_NoisyReliable,
    "PvtV2B4_FPN_MHSIU_NoisyReliable": (
        PvtV2B4_FPN_MHSIU_NoisyReliable
    ),
}
DEFAULT_MODEL = "PvtV2B4_FPN_NoisyReliable"


def apply_default_args(arguments, mode):
    defaults = {
        "--config": str(CONFIG_PATH),
        "--data-cfg": str(DATA_CONFIG_PATH),
        "--model-name": DEFAULT_MODEL,
        "--output-dir": OUTPUT_DIR,
        "--info": {
            "none": "A0_no_reliability",
            "image": "A1_noisy_q",
            "both": "A2_noisy_qU",
            "pixel": "A3_noisy_U",
        }[mode],
    }
    result = list(arguments)
    for option, value in defaults.items():
        if not any(
            item == option or item.startswith(option + "=")
            for item in result
        ):
            result.extend([option, value])
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reliability-mode",
        choices=MODES,
        default="both",
    )
    parser.add_argument("--prob-dir", default=str(PROBABILITY_DIR))
    own_args, remaining = parser.parse_known_args()
    sys.argv = [
        sys.argv[0],
        *apply_default_args(remaining, own_args.reliability_mode),
    ]

    probability_dir = Path(own_args.prob_dir)
    if not probability_dir.is_absolute():
        probability_dir = PROJECT_ROOT / probability_dir
    if own_args.reliability_mode != "none" and not probability_dir.is_dir():
        raise FileNotFoundError(
            f"Missing SAM probability directory: {probability_dir}"
        )

    core = continuous.core
    # Make both names legal argparse choices before core.parse_cfg().
    for model_name, model_class in MODEL_CLASSES.items():
        setattr(core.model_zoo, model_name, model_class)

    cfg = core.parse_cfg()
    model_class = MODEL_CLASSES.get(cfg.model_name)
    if model_class is None:
        choices = ", ".join(MODEL_CLASSES)
        raise ValueError(
            f"Unsupported reliability model {cfg.model_name!r}; "
            f"choose one of: {choices}"
        )

    # parse_cfg has already serialized the plain config. Do not access
    # cfg.pretty_text after it attaches the runtime TBLogger object.
    cfg.reliability = dict(
        mode=own_args.reliability_mode,
        probability_dir=str(probability_dir),
    )
    cfg.model = dict(cfg.get("model", {}))
    cfg.model["reliability_mode"] = own_args.reliability_mode
    core.ImageTrainDataset = partial(
        NoisyReliabilityDataset,
        probability_dir=probability_dir,
        reliability_enabled=own_args.reliability_mode != "none",
    )

    shutil.copy(__file__, cfg.path.trainer_copy)
    core.pt_utils.initialize_seed_cudnn(
        seed=cfg.base_seed,
        deterministic=cfg.deterministic,
    )
    model = model_class(
        num_frames=1,
        pretrained=cfg.pretrained,
        use_checkpoint=cfg.use_checkpoint,
        **dict(cfg.model),
    ).to(cfg.device)
    if cfg.load_from:
        core.io.load_weight(
            model=model,
            load_path=cfg.load_from,
            strict=True,
        )

    core.LOGGER.info(
        f"model={cfg.model_name} | reliability={own_args.reliability_mode} "
        f"| Clean=original BCE | probabilities={probability_dir}"
    )
    if not cfg.evaluate:
        continuous.train(model=model, cfg=cfg)
    if cfg.evaluate or cfg.has_test:
        core.io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )
        core.test(model=model, cfg=cfg)
    core.LOGGER.info("End Noisy reliability experiment")


if __name__ == "__main__":
    main()
