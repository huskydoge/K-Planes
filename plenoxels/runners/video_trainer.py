import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List

import pandas as pd
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        
        # Initialize separate metrics tracking
        self.train_metrics = []  # Store training metrics
        self.val_metrics = []    # Store validation metrics
        
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)
            
        # Paths for separate CSV files
        self.train_csv_path = os.path.join(self.log_dir, "train_metrics.csv")
        self.val_csv_path = os.path.join(self.log_dir, "validation_metrics.csv")

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)
        
        # Save training metrics to separate CSV every calc_metrics_every steps
        if self.global_step % self.calc_metrics_every == 0:
            train_metrics = {
                'step': self.global_step,
                'lr': self.lr
            }
            # Add training loss metrics
            for loss_name, loss_val in self.loss_info.items():
                train_metrics[loss_name] = loss_val.value
            
            self.train_metrics.append(train_metrics)
            self._save_train_csv()

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        # Aggregate validation metrics
        val_metrics_agg = {}
        for k in per_scene_metrics:
            val_metrics_agg[k] = np.mean(np.asarray(per_scene_metrics[k])).item()
            # Also log to tensorboard
            self.writer.add_scalar(f"test/{k}", val_metrics_agg[k], self.global_step)

        # Save validation metrics to separate CSV  
        val_metrics_record = {
            'step': self.global_step
        }
        val_metrics_record.update(val_metrics_agg)
        
        self.val_metrics.append(val_metrics_record)
        self._save_val_csv()
        
        # Log validation results
        log_text = f"step {self.global_step}/{self.num_steps}"
        for k, v in val_metrics_agg.items():
            log_text += f" | {k}: {v:.4f}"
        log.info(log_text)

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5

    def _save_train_csv(self):
        """Save training metrics to CSV file"""
        if self.train_metrics:
            df = pd.DataFrame(self.train_metrics)
            df.to_csv(self.train_csv_path, index=False)
    
    def _save_val_csv(self):
        """Save validation metrics to CSV file"""
        if self.val_metrics:
            df = pd.DataFrame(self.val_metrics)
            df.to_csv(self.val_csv_path, index=False)

    def train(self):
        """Override train method to add metrics plotting after training completion"""
        # Call original training method
        super().train()
        
        # Plot metrics after training is complete
        self._plot_metrics()
        log.info(f"Training completed! Metrics plots saved to: {self.log_dir}")

    def _plot_metrics(self):
        """Plot metrics curves"""
        if not self.train_metrics and not self.val_metrics:
            log.warning("No metrics data available for plotting")
            return
            
        # Create separate dataframes for training and validation
        train_df = pd.DataFrame(self.train_metrics) if self.train_metrics else pd.DataFrame()
        val_df = pd.DataFrame(self.val_metrics) if self.val_metrics else pd.DataFrame()
        
        # Find all metrics to plot
        train_metric_cols = [col for col in train_df.columns if col not in ['step', 'lr']]
        val_metric_cols = [col for col in val_df.columns if col not in ['step']]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot main metrics
        main_metrics = ['psnr', 'mse', 'ssim']
        
        for i, metric in enumerate(main_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot training curve
            if metric in train_df.columns:
                ax.plot(train_df['step'], train_df[metric], 
                       label=f'Train {metric.upper()}', color='blue', alpha=0.7)
            
            # Plot validation curve  
            if metric in val_df.columns:
                ax.plot(val_df['step'], val_df[metric], 
                       label=f'Val {metric.upper()}', color='red', marker='o', alpha=0.8)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} vs Training Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot learning rate curve
        if len(axes) > len(main_metrics) and 'lr' in train_df.columns:
            ax = axes[len(main_metrics)]
            ax.plot(train_df['step'], train_df['lr'], label='Learning Rate', color='green')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate vs Training Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save detailed plot with all metrics
        self._plot_all_metrics(train_df, val_df)
        
    def _plot_all_metrics(self, train_df, val_df):
        """Plot detailed chart with all metrics"""
        # Get all unique metric columns from both dataframes
        train_metric_cols = [col for col in train_df.columns if col not in ['step', 'lr']]
        val_metric_cols = [col for col in val_df.columns if col not in ['step']]
        
        # Combine all unique metric names
        all_metrics = list(set(train_metric_cols + val_metric_cols))
        metric_cols = all_metrics
        
        if not metric_cols:
            return
            
        n_cols = 3
        n_rows = (len(metric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(metric_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot training metric if available
            if col in train_df.columns:
                ax.plot(train_df['step'], train_df[col], label=f'Train {col}', color='blue')
            
            # Plot validation metric if available
            if col in val_df.columns:
                ax.plot(val_df['step'], val_df[col], label=f'Val {col}', color='red', marker='o')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(col)
            ax.set_title(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(metric_cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        # Save detailed chart
        plot_path = os.path.join(self.log_dir, 'all_metrics_detailed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def render_with_perturbations(self, param_pairs: List[tuple], reference_frames: List = None):
        """Render videos with different perturbation parameters and analyze quality metrics.
        
        Args:
            param_pairs: List of (pos_noise_scale, angle_noise) tuples
            reference_frames: Reference frames for LPIPS calculation (if None, uses ground truth)
        """
        from plenoxels.ops.image import metrics
        from plenoxels.datasets.video_datasets import generate_perturbed_render_poses
        import copy
        
        log.info(f"Starting render with {len(param_pairs)} different perturbation parameters")
        
        # Store results for analysis
        results = {}
        
        # Get reference frames (ground truth) if not provided
        if reference_frames is None:
            log.info("Rendering reference frames (ground truth)")
            reference_frames = []
            
            # Use the original test dataset directly for reference (unperturbed)
            from plenoxels.datasets.video_datasets import load_dmlab_data
            poses, imgs, timestamps, per_cam_near_fars = load_dmlab_data(
                self.test_dataset.datadir, split='test', max_tsteps=self.test_dataset.max_tsteps)
            
            # Store original poses
            original_poses = self.test_dataset.poses.clone()
            
            # Temporarily set unperturbed poses
            self.test_dataset.poses = poses.float()
            
            pb_ref = tqdm(total=len(self.test_dataset.poses), desc="Rendering reference")
            for img_idx, data in enumerate(self.test_dataset):
                preds = self.eval_step(data)
                img_h, img_w = self.test_dataset.img_h, self.test_dataset.img_w
                pred_rgb = preds["rgb"].reshape(img_h, img_w, 3).cpu().clamp(0, 1).mul(255.0).byte().numpy()
                reference_frames.append(pred_rgb)
                pb_ref.update(1)
            pb_ref.close()
            
            # Restore original poses
            self.test_dataset.poses = original_poses
        
        # Render with different perturbation parameters
        for pos_noise, angle_noise in param_pairs:
            param_name = f"pos{pos_noise:.4f}_angle{angle_noise:.2f}"
            log.info(f"Rendering with perturbation: {param_name}")
            
            # Load original test data
            poses, imgs, timestamps, per_cam_near_fars = load_dmlab_data(
                self.test_dataset.datadir, split='test', max_tsteps=self.test_dataset.max_tsteps)
            
            # Generate perturbed poses
            perturbed_poses = generate_perturbed_render_poses(
                poses.numpy(), n_variations=1, 
                pos_noise_scale=pos_noise, angle_noise=angle_noise)
            
            # Store original poses and temporarily set perturbed poses
            original_poses = self.test_dataset.poses.clone()
            self.test_dataset.poses = torch.from_numpy(perturbed_poses).float()
            
            # Render frames
            pred_frames = []
            lpips_scores = []
            
            pb = tqdm(total=len(self.test_dataset.poses), desc=f"Rendering {param_name}")
            for img_idx, data in enumerate(self.test_dataset):
                preds = self.eval_step(data)
                img_h, img_w = self.test_dataset.img_h, self.test_dataset.img_w
                pred_rgb = preds["rgb"].reshape(img_h, img_w, 3).cpu().clamp(0, 1).mul(255.0).byte().numpy()
                pred_frames.append(pred_rgb)
                
                # Calculate LPIPS for this frame
                if img_idx < len(reference_frames):
                    pred_tensor = torch.from_numpy(pred_rgb).float() / 255.0  # [H, W, 3]
                    ref_tensor = torch.from_numpy(reference_frames[img_idx]).float() / 255.0  # [H, W, 3]
                    
                    # rgb_lpips expects [H, W, 3] format
                    lpips_score = metrics.rgb_lpips(pred_tensor, ref_tensor, device=pred_tensor.device)
                    lpips_scores.append(lpips_score)
                
                pb.update(1)
            pb.close()
            
            # Restore original poses
            self.test_dataset.poses = original_poses
            
            # Calculate average LPIPS
            avg_lpips = np.mean(lpips_scores) if lpips_scores else 0.0
            
            # Save video
            video_filename = f"perturbation_{param_name}.mp4"
            video_path = os.path.join(self.log_dir, video_filename)
            write_video_to_file(video_path, pred_frames)
            
            # Store results
            results[param_name] = {
                'pos_noise_scale': pos_noise,
                'angle_noise': angle_noise,
                'avg_lpips': avg_lpips,
                'video_path': video_path,
                'frames': pred_frames
            }
            
            log.info(f"Saved {param_name}: LPIPS = {avg_lpips:.4f}")
        
        # Plot analysis charts
        self._plot_perturbation_analysis(results)
        
        # Save results summary
        self._save_perturbation_results(results)
        
        return results
    
    def _plot_perturbation_analysis(self, results: dict):
        """Plot relationship between perturbation parameters and LPIPS scores"""
        
        # Extract data for plotting
        pos_noise_scales = [data['pos_noise_scale'] for data in results.values()]
        angle_noises = [data['angle_noise'] for data in results.values()]
        lpips_scores = [data['avg_lpips'] for data in results.values()]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Position noise vs LPIPS
        # Group by angle_noise to show different series
        angle_groups = {}
        for i, (pos, angle, lpips) in enumerate(zip(pos_noise_scales, angle_noises, lpips_scores)):
            if angle not in angle_groups:
                angle_groups[angle] = {'pos': [], 'lpips': []}
            angle_groups[angle]['pos'].append(pos)
            angle_groups[angle]['lpips'].append(lpips)
        
        for angle, data in angle_groups.items():
            ax1.plot(data['pos'], data['lpips'], 'o-', label=f'Angle noise: {angle:.2f}°', alpha=0.7)
        
        ax1.set_xlabel('Position Noise Scale')
        ax1.set_ylabel('Average LPIPS')
        ax1.set_title('Position Perturbation vs Image Quality')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Angle noise vs LPIPS
        # Group by pos_noise_scale to show different series
        pos_groups = {}
        for i, (pos, angle, lpips) in enumerate(zip(pos_noise_scales, angle_noises, lpips_scores)):
            if pos not in pos_groups:
                pos_groups[pos] = {'angle': [], 'lpips': []}
            pos_groups[pos]['angle'].append(angle)
            pos_groups[pos]['lpips'].append(lpips)
        
        for pos, data in pos_groups.items():
            ax2.plot(data['angle'], data['lpips'], 's-', label=f'Pos noise: {pos:.4f}', alpha=0.7)
        
        ax2.set_xlabel('Angle Noise (degrees)')
        ax2.set_ylabel('Average LPIPS')
        ax2.set_title('Rotation Perturbation vs Image Quality')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, 'perturbation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Perturbation analysis plot saved to: {plot_path}")
    
    def _save_perturbation_results(self, results: dict):
        """Save perturbation results to CSV file"""
        
        # Prepare data for CSV
        csv_data = []
        for param_name, data in results.items():
            csv_data.append({
                'parameter_name': param_name,
                'pos_noise_scale': data['pos_noise_scale'],
                'angle_noise_degrees': data['angle_noise'],
                'avg_lpips': data['avg_lpips'],
                'video_filename': os.path.basename(data['video_path'])
            })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.log_dir, 'perturbation_results.csv')
        df.to_csv(csv_path, index=False)
        
        log.info(f"Perturbation results saved to: {csv_path}")


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, data_downsample = 2.0, **kwargs):
    downsample = data_downsample # Both D-NeRF and DyNeRF use downsampling by 2
    ts_dset = Video360Dataset(
        data_dir, split=split, downsample=downsample,
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
    )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, data_downsample=data_downsample, **kwargs))
    return od
