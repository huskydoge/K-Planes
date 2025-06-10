# https://github.com/sarafridov/K-Planes/issues/1
config = {
 'expname': 'dmlab_debug_train_fr_50_test_fr_50',
 'logdir': './logs/dmlab',
 'device': 'cuda:0',

 # Data settings
 'data_downsample': 1.0,  # DMlab data is already 128x128, no need to downsample
 'data_dirs': ['/data/hansen/projects/benhao/wm-memory/data/dmlab/debug/1'],
 'contract': False,
 'ndc': False,  # DMlab uses standard perspective projection
 'max_train_tsteps': 50,  
 'max_test_tsteps': 50,  
 'keyframes': True,
 'scene_bbox': [[92.9, 86.5, 1.1], [712.6, 600.0, 101.1]],  # DMlab room dimensions
 
 # Importance sampling settings
 'isg': False,  # Start without ISG, will be enabled automatically when weights are computed
 'isg_step': -1,  # -1 means don't use ISG initially
 'ist_step': -1,  # Disable IST - not suitable for ego-centric camera motion

 # Optimization settings

 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.001,
 'time_smoothness_weight_proposal_net': 1e-05,

 # Training settings
 'save_every': 1000,
 'valid_every': 1000,
 'save_outputs': True,
 'train_fp16': True,
 'num_steps': 10000,  # Reduced for initial testing
 'batch_size': 8192,  # Smaller batch for 128x128 images
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_samples': [128, 64],  # Reduced for smaller scene
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [96, 96, 96, 100]},   # Adjusted for DMlab scene
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [192, 192, 192, 100]}
 ],

 # Model settings
 'concat_features_across_scales': True,
 'density_activation': 'trunc_exp',
 'linear_decoder': True,
 'linear_decoder_layers': 1,
 'multiscale_res': [1, 2, 4],  # Reduced number of scales for smaller scene
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,  # 3D + time
  'output_coordinate_dim': 32,
  'resolution': [48, 48, 48, 100]  # Adjusted for DMlab scene size and temporal resolution
 }],
}
