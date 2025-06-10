import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any
import tempfile

import numpy as np
import torch
import torch.utils.data
from plenoxels.runners import video_trainer
from plenoxels.runners import phototourism_trainer
from plenoxels.runners import static_trainer
from plenoxels.utils.create_rendering import render_to_path, decompose_space_time
from plenoxels.utils.parse_args import parse_optfloat
from omegaconf import OmegaConf
from pathlib import Path


def get_freer_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fname = os.path.join(tmpdir, "tmp")
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >"{tmp_fname}"')
        if os.path.isfile(tmp_fname):
            memory_available = [int(x.split()[2]) for x in open(tmp_fname, 'r').readlines()]
            if len(memory_available) > 0:
                return np.argmax(memory_available)
    return None  # The grep doesn't work with all GPUs. If it fails we ignore it.

gpu = get_freer_gpu()
if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu}")
else:
    print(f"Did not set GPU.")

import torch
import torch.utils.data
from plenoxels.runners import video_trainer
from plenoxels.runners import phototourism_trainer
from plenoxels.runners import static_trainer
from plenoxels.utils.create_rendering import render_to_path, decompose_space_time
from plenoxels.utils.parse_args import parse_optfloat


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        return phototourism_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs
        )
    else:
        return static_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)


def init_trainer(model_type: str, **kwargs):
    if model_type == "video":
        from plenoxels.runners import video_trainer
        return video_trainer.VideoTrainer(**kwargs)
    elif model_type == "phototourism":
        from plenoxels.runners import phototourism_trainer
        return phototourism_trainer.PhototourismTrainer(**kwargs)
    else:
        from plenoxels.runners import static_trainer
        return static_trainer.StaticTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--perturbation-analysis', action='store_true', help='Run perturbation analysis')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    
    # Process overrides from argparse into config
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    
    # Determine model type
    if "keyframes" in config:
        model_type = "video"
    elif "appearance_embedding_dim" in config:
        model_type = "phototourism"
    else:
        model_type = "static"
        
    render_only = args.render_only
    validate_only = args.validate_only
    spacetime_only = args.spacetime_only
    perturbation_analysis = args.perturbation_analysis

    # For validation, rendering, or analysis modes, ensure we have log_dir
    if validate_only or render_only or spacetime_only or perturbation_analysis:
        if args.log_dir is not None:
            config['logdir'] = args.log_dir
        else:
            raise ValueError("--log-dir is required for validation, rendering, or analysis modes")

    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    # Print config for debugging
    pprint.pprint(config)
    
    # Save config if training
    if not (validate_only or render_only or spacetime_only or perturbation_analysis):
        save_config(config)
    
    data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only or perturbation_analysis, **config)
    config.update(data)
    
    trainer = init_trainer(model_type, **config)

    if validate_only:
        checkpoint_path = os.path.join(config["logdir"], "model.pth")
        trainer.load_model(torch.load(checkpoint_path), training_needed=False)
        trainer.validate()
    elif render_only:
        checkpoint_path = os.path.join(config["logdir"], "model.pth")
        trainer.load_model(torch.load(checkpoint_path), training_needed=False)
        render_to_path(trainer)
    elif spacetime_only:
        checkpoint_path = os.path.join(config["logdir"], "model.pth")
        trainer.load_model(torch.load(checkpoint_path), training_needed=False)
        decompose_space_time(trainer)
    elif perturbation_analysis:
        checkpoint_path = os.path.join(config["logdir"], "model.pth")
        trainer.load_model(torch.load(checkpoint_path), training_needed=False)
        # Define parameter pairs for analysis
        param_pairs = [
            (0.0, 0.0),      # Baseline
            (0.001, 0.0),    # Small position noise
            (0.005, 0.0),    # Medium position noise
            (0.01, 0.0),     # Large position noise
            (0.0, 0.1),      # Small rotation noise
            (0.0, 0.5),      # Medium rotation noise
            (0.0, 1.0),      # Large rotation noise
            (0.005, 0.5),    # Combined noise
        ]
        trainer.render_with_perturbations(param_pairs)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
