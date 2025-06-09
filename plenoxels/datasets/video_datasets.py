import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        elif "dmlab" in datadir:
            dset_type = "dmlab"
        else:
            dset_type = "llff"

        # Note: timestamps are stored normalized between -1, 1.
        if dset_type == "llff":
            if split == "render":
                assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                per_cam_poses, per_cam_near_fars, intrinsics, _ = load_llffvideo_poses(
                    datadir, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                render_poses = generate_spiral_path(
                    per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=300,
                    n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                self.poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                timestamps = torch.linspace(0, 299, len(self.poses))
                imgs = None
            else:
                per_cam_poses, per_cam_near_fars, intrinsics, videopaths = load_llffvideo_poses(
                    datadir, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
                if split == 'test':
                    keyframes = False
                poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                    videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics,
                    split=split, keyframes=keyframes, keyframes_take_each=30)
                self.poses = poses.float()
                if contraction:
                    self.per_cam_near_fars = per_cam_near_fars.float()
                else:
                    self.per_cam_near_fars = torch.tensor(
                        [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)
            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0, 0, 2.])
            self.global_scale = torch.tensor([0.5, 0.6, 1])
            # Normalize timestamps between -1, 1
            timestamps = (timestamps.float() / 299) * 2 - 1
        elif dset_type == "synthetic":
            assert not contraction, "Synthetic video dataset does not work with contraction."
            assert not ndc, "Synthetic video dataset does not work with NDC."
            if split == 'render':
                num_tsteps = 120
                dnerf_durations = {'hellwarrior': 100, 'mutant': 150, 'hook': 100, 'bouncingballs': 150, 'lego': 50, 'trex': 200, 'standup': 150, 'jumpingjacks': 200}
                for scene in dnerf_durations.keys():
                    if 'dnerf' in datadir and scene in datadir:
                        num_tsteps = dnerf_durations[scene]
                render_poses = torch.stack([
                    generate_spherical_poses(angle, -30.0, 4.0)
                    for angle in np.linspace(-180, 180, num_tsteps + 1)[:-1]
                ], 0)
                imgs = None
                self.poses = render_poses
                timestamps = torch.linspace(0.0, 1.0, render_poses.shape[0])
                _, transform = load_360video_frames(
                    datadir, 'train', max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                img_h, img_w = 800, 800
            else:
                frames, transform = load_360video_frames(
                    datadir, split, max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                imgs, self.poses = load_360_images(frames, datadir, split, self.downsample)
                timestamps = torch.tensor(
                    [fetch_360vid_info(f)[0] for f in frames], dtype=torch.float32)
                img_h, img_w = imgs[0].shape[:2]
            if ndc:
                self.per_cam_near_fars = torch.tensor([[0.0, self.ndc_far]])
            else:
                self.per_cam_near_fars = torch.tensor([[2.0, 6.0]])
            if "dnerf" in datadir:
                # dnerf time is between 0, 1. Normalize to -1, 1
                timestamps = timestamps * 2 - 1
            else:
                # lego (our vid) time is like dynerf: between 0, 30.
                timestamps = (timestamps.float() / torch.amax(timestamps)) * 2 - 1
            intrinsics = load_360_intrinsics(
                transform, img_h=img_h, img_w=img_w, downsample=self.downsample)
        elif dset_type == "dmlab":
            if split == "render":
                # Load data and create render poses
                poses, imgs, timestamps, per_cam_near_fars = load_dmlab_data(datadir, split='train', max_tsteps=self.max_tsteps)
                # Generate spiral render path around the scene
                render_poses = generate_dmlab_render_poses(poses.numpy(), n_frames=120)
                self.poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = per_cam_near_fars[:1]  # Use first camera's near/far
                timestamps = torch.linspace(0, 119, len(self.poses))
                imgs = None
            else:
                poses, imgs, timestamps, per_cam_near_fars = load_dmlab_data(datadir, split=split, max_tsteps=self.max_tsteps)
                self.poses = poses.float()
                self.per_cam_near_fars = per_cam_near_fars.float()
            intrinsics = load_dmlab_intrinsics(img_h=128, img_w=128, fov_degrees=90.0, downsample=self.downsample)
            self.global_translation = torch.tensor([0, 0, 0])
            self.global_scale = torch.tensor([1, 1, 1])
            # Normalize timestamps between -1, 1
            timestamps = (timestamps.float() / (timestamps.max() + 1e-6)) * 2 - 1
        else:
            raise ValueError(datadir)

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)
        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        elif imgs is not None:
            # Correct reshaping: preserve frame structure [n_frames, height*width, 3]
            imgs = imgs.view(imgs.shape[0], -1, imgs.shape[-1])

        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = int(4 / downsample)
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        self.isg_weights = None
        self.ist_weights = None
        if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
            if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
                log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
            else:
                # Precompute ISG weights
                t_s = time.time()
                gamma = 1e-3 if self.keyframes else 2e-2
                self.isg_weights = dynerf_isg_weight(
                    imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                    median_imgs=self.median_imgs, gamma=gamma)
                # Normalize into a probability distribution, to speed up sampling
                self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                torch.save(self.isg_weights, os.path.join(datadir, f"isg_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

            if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                self.ist_weights = torch.load(os.path.join(datadir, f"ist_weights.pt"))
                log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
            else:
                # Precompute IST weights
                t_s = time.time()
                self.ist_weights = dynerf_ist_weight(
                    imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                    num_cameras=self.median_imgs.shape[0])
                # Normalize into a probability distribution, to speed up sampling
                self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                torch.save(self.ist_weights, os.path.join(datadir, f"ist_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

        if self.isg:
            self.enable_isg()

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }
        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            # For test/validation, use the near_far bounds corresponding to current image
            frame_id = index // (h * w) if self.split == 'test' else 0
            frame_id = min(frame_id, len(self.per_cam_near_fars) - 1)  # Clamp to available frames
            out['near_fars'] = self.per_cam_near_fars[frame_id:frame_id+1]  # Keep 2D shape [1, 2]

        if self.imgs is not None:
            if self.split == 'train':
                out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])
            else:
                # For test split, imgs is organized as [n_frames, height*width, 3]
                # index is the frame number, we need all pixels of that frame
                frame_id = index
                out['imgs'] = (self.imgs[frame_id] / 255.0)  # Already [height*width, 3]

        c2w = self.poses[image_id]                                    # [num_rays or 1, 3, 4]
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=True)                                        # [num_rays, 3]

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        return out


def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    elif dset_type == 'dmlab':
        # DMlab specific bounding box based on typical room dimensions
        # From analysis: X:[143-663], Y:[137-550], Z:[51] with margin
        return torch.tensor([[92.9, 86.5, 1.1], [712.6, 600.0, 101.1]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])
        log.info(f"Selected subset of {len(pose_ids)} camera poses: {pose_ids}.")

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])
    return sub_frames, meta


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0]) # Benhao: use first one for test
    else:
        split_ids = np.arange(poses.shape[0])
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)
    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    """
    Computes ISG (Importance-driven Scene-adaptive Geometry-sampling) weights.
    This weight measures how much each pixel deviates from the static background (median image).
    Formula (from DyNeRF paper):
        ψ(p,t) = E(p,t)² / (E(p,t)² + γ²)
        where E(p,t) = I(p,t) - I_M(p) is the difference between the current frame and the median image.
    """
    # imgs: [num_cameras * num_frames, H, W, 3]
    # median_imgs: [num_cameras, H, W, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape

    # 1. Compute the difference E(p,t) = I(p,t) - I_M(p)
    #    - Reshape imgs to [num_cameras, num_frames, H, W, C]
    #    - Normalize pixel values from [0, 255] to [0, 1]
    #    - median_imgs [num_cameras, 1, H, W, C] is broadcast over all frames for subtraction.
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # Compute the squared difference E(p,t)²
    )  # [num_cameras, num_frames, H, W, 3]

    # 2. Apply the formula ψ = E² / (E² + γ²)
    #    This makes weights for static regions approach 0 and for dynamic regions approach 1.
    psidiff = squarediff.div_(squarediff + gamma**2)

    # 3. Average over the three RGB channels to get a single-channel weight map.
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # Returns weights in the range [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    """
    Computes IST (Importance-driven Scene-adaptive Temporal-sampling) weights.
    This weight measures the intensity of temporal changes for each pixel by comparing it with adjacent frames.
    Formula (from DyNeRF paper):
        τ(p,t) = max( max_{s∈[1,S]} |I(p,t) - I(p, t±s)|, α )
    """
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    # Reshape input to [num_cameras, num_timesteps, H, W, 3]
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    # Define the temporal window, s from 1 to frame_shift
    shifts = list(range(frame_shift + 1))[1:]
    # Iterate over all temporal shifts s
    for shift in shifts:
        # 1. Create videos shifted forwards and backwards in time by s frames.
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)

        # 2. Compute the absolute color difference |I(p,t) - I(p,t±s)| between the current frame
        #    and the shifted frames, and take the maximum of the two.
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))

        # 3. Accumulate the maximum difference found across the entire temporal window S.
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]

    # 4. Average over the RGB color channels.
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]

    # 5. Apply alpha as a lower bound for the weights, ensuring all pixels have a chance to be sampled.
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff


# ========================================
# DMlab Dataset Functions
# ========================================

def load_dmlab_data(datadir: str, split: str, max_tsteps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load DMlab data from npz file.
    
    Args:
        datadir (str): Directory containing the npz file
        split (str): 'train' or 'test'
        max_tsteps (Optional[int]): Maximum number of timesteps to load
    
    Returns:
        Tuple of:
        - poses: [N, 3, 4] camera-to-world transformation matrices
        - imgs: [N, H, W, 3] RGB images in range [0, 1]
        - timestamps: [N] frame indices
        - near_fars: [N, 2] near and far bounds for each frame
    """
    # Find npz file in the directory
    npz_files = glob.glob(os.path.join(datadir, "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {datadir}")
    
    data_path = npz_files[0]  # Use first npz file
    log.info(f"Loading DMlab data from: {data_path}")
    
    data = np.load(data_path)
    
    # Load video data: (T, H, W, C) uint8 [0, 255]
    video = data['video']  # (T, H, W, 3)
    num_frames = video.shape[0]
    
    # Load camera poses and rotations
    camera_pos = data['camera_pos'].reshape(-1, 3)  # (T, 3)
    
    # Use quaternion rotation if available, otherwise fall back to euler angles
    if 'rot' in data and data['rot'].shape[1] == 4:
        camera_rot = data['rot']  # (T, 4) quaternions [w, x, y, z]
    else:
        camera_rot = data['camera_rot'].reshape(-1, 3)  # (T, 3) in degrees
    
    # Load projection matrices for near/far extraction
    proj_matrices = data['proj_matrices']  # (T, 4, 4)
    
    # Load depth if available (Note: DMlab depth is reverse NDC depth)
    # depth_video = data['depth_video']  # (T, H, W, 1) - zfar=0, znear=1
    
    if max_tsteps is not None and max_tsteps < num_frames:
        # Subsample frames
        step = num_frames // max_tsteps
        indices = np.arange(0, num_frames, step)[:max_tsteps]
        log.info(f"Subsampling DMlab data: {num_frames} -> {len(indices)} frames")
        
        video = video[indices]
        camera_pos = camera_pos[indices]
        camera_rot = camera_rot[indices]
        proj_matrices = proj_matrices[indices]
        num_frames = len(indices)
    
    # Create train/test split
    if split == 'train':
        # Use first 70% for training (ensure sufficient frames)
        split_point = max(1, int(0.7 * num_frames))
        frame_indices = np.arange(0, split_point)
    elif split == 'test':
        # Use last 30% for testing (ensure sufficient frames for video)
        split_point = max(1, int(0.7 * num_frames))
        frame_indices = np.arange(split_point, num_frames)
    else:
        # Use all frames
        frame_indices = np.arange(num_frames)
    
    # Apply split
    video = video[frame_indices]
    camera_pos = camera_pos[frame_indices]
    camera_rot = camera_rot[frame_indices]
    proj_matrices = proj_matrices[frame_indices]
    
    # Convert images to [0, 1] range
    imgs = torch.from_numpy(video.astype(np.float32) / 255.0)  # [N, H, W, 3]
    
    # Convert camera poses
    poses = dmlab_poses_to_c2w(camera_pos, camera_rot)  # [N, 3, 4]
    
    # Extract near/far bounds from projection matrices
    near_fars = extract_near_far_from_projection(proj_matrices)  # [N, 2]
    
    # Create timestamps
    timestamps = torch.arange(len(frame_indices), dtype=torch.float32)
    
    log.info(f"Loaded DMlab {split} data: {len(imgs)} frames, "
             f"image size: {imgs.shape[1]}x{imgs.shape[2]}, "
             f"pose range: {poses.min():.3f} to {poses.max():.3f}")
    
    return poses, imgs, timestamps, near_fars


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


def calculate_c2w(pos, rot):
    """
    Camera2world matrix

    pos: (T, 3), the camera position under world coordinate
    rot: (T, 4), the camera rotation under world coordinate (quaternion [w,x,y,z])
    """
    camera_matrices = []
    
    for i in range(len(pos)):
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(rot[i])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        # Build c2w matrix. Use transpose for world-to-camera to camera-to-world conversion.
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        if pos[i].ndim == 1:
            c2w[:3, 3] = np.array([pos[i][0], pos[i][1], pos[i][2]])
        else:
            c2w[:3, 3] = np.array([pos[i][0][0], pos[i][0][1], pos[i][0][2]])
        
        camera_matrices.append(c2w)
    
    camera_matrices = np.array(camera_matrices, dtype=np.float32)  # (T, 4, 4)
    return torch.tensor(camera_matrices, dtype=torch.float32)  # (T, 4, 4)


def dmlab_poses_to_c2w(camera_pos: np.ndarray, camera_rot: np.ndarray) -> torch.Tensor:
    """Convert DMlab camera position and rotation to camera-to-world matrices.
    
    Args:
        camera_pos: [N, 3] camera positions in world coordinates
        camera_rot: [N, 3] camera rotations in degrees (pitch, yaw, roll) OR [N, 4] quaternions
    
    Returns:
        [N, 3, 4] camera-to-world transformation matrices
    """
    if camera_rot.shape[1] == 4:
        # Quaternion format
        c2w_matrices = calculate_c2w(camera_pos, camera_rot)
        return c2w_matrices[:, :3, :]  # Return [N, 3, 4]
    else:
        # Euler angles format - fallback to original implementation
        import math
        
        N = camera_pos.shape[0]
        c2w_matrices = np.zeros((N, 3, 4), dtype=np.float32)
        
        for i in range(N):
            pos = camera_pos[i]
            rot = camera_rot[i]
            
            # Convert degrees to radians
            pitch = math.radians(rot[0])  # rotation around X axis
            yaw = math.radians(rot[1])    # rotation around Y axis  
            roll = math.radians(rot[2])   # rotation around Z axis
            
            # Create rotation matrices
            # Rotation order: yaw -> pitch -> roll (Y -> X -> Z)
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            
            # Combined rotation matrix (YXZ order)
            R = np.array([
                [cos_y*cos_r - sin_y*sin_p*sin_r, -cos_y*sin_r - sin_y*sin_p*cos_r, -sin_y*cos_p],
                [cos_p*sin_r, cos_p*cos_r, -sin_p],
                [sin_y*cos_r + cos_y*sin_p*sin_r, -sin_y*sin_r + cos_y*sin_p*cos_r, cos_y*cos_p]
            ], dtype=np.float32)
            
            # Create c2w matrix
            c2w_matrices[i, :3, :3] = R
            c2w_matrices[i, :3, 3] = pos
        
        return torch.from_numpy(c2w_matrices)


def extract_znear_zfar_from_projection(Ps):
    """
    Extracts near and far plane distances from a right-handed projection matrix.
    NOTE: The formula B/(A+1) yields the far plane and B/(A-1) yields the near plane.
    This function returns them in that "swapped" order, as required by the
    linearization function for a reversed-Z depth buffer.
    """
    # Using names that reflect the swapped output needed for the next step.
    far_distances = []
    near_distances = []
    for P in Ps:
        A = P[2, 2]
        B = P[2, 3]

        # In a right-handed system, B/(A+1) is the far plane distance.
        far_dist = B / (A + 1.0)
        # And B/(A-1) is the near plane distance.
        near_dist = B / (A - 1.0)
        
        far_distances.append(far_dist)
        near_distances.append(near_dist)
    
    # Aggregate and return in the order the linearization function expects.
    # The first returned value is the FAR distance, the second is the NEAR distance.
    # return np.min(far_distances), np.max(near_distances)
    return np.array(far_distances), np.array(near_distances)


def extract_near_far_from_projection(proj_matrices: np.ndarray) -> torch.Tensor:
    """Extract near and far planes from projection matrices.
    
    Args:
        proj_matrices: [N, 4, 4] perspective projection matrices
    
    Returns:
        [N, 2] near and far bounds for each frame
    """
    far_distances, near_distances = extract_znear_zfar_from_projection(proj_matrices)
    # Stack as [N, 2] where each row is [near, far]
    near_fars = np.stack([near_distances, far_distances], axis=1)
    return torch.from_numpy(near_fars.astype(np.float32))


def load_dmlab_intrinsics(img_h: int, img_w: int, fov_degrees: float, downsample: float) -> Intrinsics:
    """Create camera intrinsics for DMlab data.
    
    Args:
        img_h: Image height
        img_w: Image width  
        fov_degrees: Field of view in degrees
        downsample: Downsampling factor
    
    Returns:
        Camera intrinsics object
    """
    from .intrinsics import Intrinsics
    import math
    
    # Compute downsampled dimensions
    height = int(img_h / downsample)
    width = int(img_w / downsample)
    
    # Compute focal length from FOV
    # For square FOV (fovx = fovy = 90 degrees)
    fov_rad = math.radians(fov_degrees)
    focal_length = (width / 2.0) / math.tan(fov_rad / 2.0)
    
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    
    return Intrinsics(
        height=height,
        width=width,
        focal_x=focal_length,
        focal_y=focal_length,
        center_x=cx,
        center_y=cy
    )


def generate_dmlab_render_poses(train_poses: np.ndarray, n_frames: int = 120) -> np.ndarray:
    """Generate render poses for DMlab data.
    
    Args:
        train_poses: [N, 3, 4] training camera poses
        n_frames: Number of render frames to generate
    
    Returns:
        [n_frames, 3, 4] render poses
    """
    # Extract camera positions and compute scene center
    positions = train_poses[:, :3, 3]  # [N, 3]
    scene_center = positions.mean(axis=0)
    
    # Compute average camera height and radius from center
    avg_height = positions[:, 2].mean()
    radius = np.linalg.norm(positions[:, :2] - scene_center[:2], axis=1).mean()
    
    # Generate circular path around scene center
    render_poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        # Camera position on circular path
        pos = np.array([
            scene_center[0] + radius * np.cos(angle),
            scene_center[1] + radius * np.sin(angle), 
            avg_height
        ])
        
        # Camera looks towards scene center
        forward = scene_center - pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up vector
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create pose matrix
        pose = np.eye(4)[:3]  # [3, 4]
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # Camera looks along -Z
        pose[:3, 3] = pos
        
        render_poses.append(pose)
    
    return np.stack(render_poses, 0)


def getRawDepth(depth_buffer_val, proj_matrices):
    """Converts depth from buffer a [0,1] range to view-space linear depth."""
    zNears, zFars = extract_znear_zfar_from_projection(proj_matrices)
    while zNears.ndim < depth_buffer_val.ndim:
        zNears = np.expand_dims(zNears, -1)
        zFars = np.expand_dims(zFars, -1)
    ndc_depth = depth_buffer_val * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    view_space_depth = (2.0 * zNears * zFars) / (zFars + zNears - ndc_depth * (zFars - zNears))
    return view_space_depth


def convert_dmlab_depth(depth: np.ndarray, proj_matrices: np.ndarray) -> np.ndarray:
    """Convert DMlab reverse NDC depth to view-space linear depth.
    
    DMlab depth format:
    - Value 0 corresponds to zfar (far plane)
    - Value 1 corresponds to znear (near plane)
    
    Args:
        depth: [..., H, W] or [..., H, W, 1] reverse NDC depth values in [0, 1]
        proj_matrices: [..., 4, 4] projection matrices
    
    Returns:
        [..., H, W] view-space linear depth values
    """
    return getRawDepth(depth, proj_matrices)
