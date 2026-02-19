import json
import logging
from pathlib import Path
import pandas as pd

import torch
from colorlog import ColoredFormatter

from av_bench.args import get_eval_parser
from av_bench.evaluate import evaluate
from av_bench.extract import extract

log = logging.getLogger()
device = "cuda"

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


setup_eval_logging()


@torch.inference_mode()
def main(args):
    gt_audio: Path = args.gt_audio
    gt_cache: Path = args.gt_cache
    pred_audio: Path = args.pred_audio
    pred_cache: Path = args.pred_cache
    audio_length: float = args.audio_length
    num_workers: int = args.num_workers
    gt_batch_size: int = args.gt_batch_size
    pred_batch_size: int = args.pred_batch_size
    unpaired: bool = args.unpaired
    skip_video_related: bool = args.skip_video_related
    skip_clap: bool = args.skip_clap
    recompute_gt_cache: bool = args.recompute_gt_cache
    recompute_pred_cache: bool = args.recompute_pred_cache

    # apply default path
    if gt_cache is None:
        if gt_audio is None:
            raise ValueError("Must specify either gt_audio or gt_cache")
        gt_cache = gt_audio / "cache"
        log.info(f"No gt cache specified, using default {gt_cache}")
        log.info(
            f"NOTE: If you are evaluating on video datasets, you must extract video cache separately"
            + f" via extract_video.py. Otherwise video-related scores will be skipped."
        )

    if pred_cache is None:
        if pred_audio is None:
            raise ValueError("Must specify either pred_audio or pred_cache")
        pred_cache = pred_audio / "cache"
        log.info(f"No pred cache specified, using default {pred_cache}")

    gt_cache = gt_cache.expanduser()
    pred_cache = pred_cache.expanduser()

    log.info(f"GT cache path: {gt_cache}")
    log.info(f"Pred cache: {pred_cache}")
    log.info(f"Audio length: {audio_length}")
    log.info(f"Unpaired: {unpaired}")

    # extract for GT if needed
    if not (gt_cache / "passt_features_embed.pth").exists() or recompute_gt_cache:
        log.info("Extracting GT cache...")
        if gt_audio is None:
            raise ValueError("Must specify gt_audio to compute gt_cache")
        gt_audio = gt_audio.expanduser()
        log.info(f"GT audio path: {gt_audio}")
        extract(
            audio_path=gt_audio,
            output_path=gt_cache,
            audio_length=audio_length,
            device=device,
            batch_size=gt_batch_size,
            num_workers=num_workers,
            skip_video_related=True,
            skip_clap=True,
        )

    # extract for pred if needed
    if not (pred_cache / "passt_features_embed.pth").exists() or recompute_pred_cache:
        log.info("Extracting pred cache...")
        if pred_audio is None:
            raise ValueError("Must specify pred_audio to compute pred_cache")
        pred_audio = pred_audio.expanduser()
        log.info(f"Pred audio path: {pred_audio}")
        extract(
            audio_path=pred_audio,
            output_path=pred_cache,
            audio_length=audio_length,
            device=device,
            batch_size=pred_batch_size,
            num_workers=num_workers,
            skip_video_related=skip_video_related,
            skip_clap=skip_clap,
        )

    log.info("Starting evaluation...")
    gt_set_p = None
    gt_set = None

    if audio_length == 8:
        gt_set_p = "/home/khaled.koutini/workspaces/MMAudio/8s_file_list.csv"
    if audio_length == 5:
        gt_set_p = "/home/khaled.koutini/workspaces/MMAudio/5s_file_list.csv"

    if "ogamedata" in str(gt_cache) or "foleybench" in str(gt_cache):
        gt_set_p = None
    if gt_set_p is not None:
        log.info(f"Using ground truth set file at {gt_set_p} for evaluation")
        df = pd.read_csv(gt_set_p, header=None)
        gt_set = set(df[0])
        log.info(f"Number of samples in ground truth set: {len(gt_set)}")
    num_samples = 1
    output_metrics = evaluate(
        gt_audio_cache=gt_cache,
        pred_audio_cache=pred_cache,
        num_samples=num_samples,
        is_paired=not unpaired,
        gt_set=gt_set,
    )

    for k, v in output_metrics.items():
        # pad k to 10 characters
        # pad v to 10 decimal places
        log.info(f"{k:<10}: {v:.10f}")

    # write output metrics to file
    output_metrics_file = pred_cache / "output_metrics.json"
    with open(output_metrics_file, "w") as f:
        json.dump(output_metrics, f, indent=4)
    log.info(f"Output metrics written to {output_metrics_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_eval_parser().parse_args()
    main(args)
