_base_=['../_base_/losses/all_losses.py',
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',

       '../_base_/datasets/nyu.py',
       '../_base_/datasets/kubric.py',

        '../_base_/default_runtime.py',
       '../_base_/schedules/schedule_1m.py',
       ]

model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=4,
        n_downsample=2,
        detach=False,
    ),
)

# loss method
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', sample_ratio=0.2, loss_weight=0.1),   
        dict(type='GRUSequenceLoss', loss_weight=1.0, loss_gamma=0.9, stereo_sup=0),
        dict(type='DeNoConsistencyLoss', loss_weight=0.001, loss_fn='CEL', scale=2)
    ],
)

data_array = [
     [
          dict(Kubric='Kubric_dataset'),
     ],
]


# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
#     crop_size=(544, 1216),
#     crop_size = (544, 992),
    crop_size = (616, 1064),  # %28 = 0

    # added from https://github.com/YvanYin/Metric3D/issues/103
    clip_depth_range=(0.1, 200),
) 

# online evaluation
# evaluation = dict(online_eval=True, interval=1000, metrics=['abs_rel', 'delta1', 'rmse'], multi_dataset_eval=True)
#log_interval = 100

interval = 100
log_interval = 10
evaluation = dict(
    online_eval=True, 
    interval=interval, 
    metrics=['abs_rel', 'mae', 'delta1', 'rmse', 'normal_mean', 'normal_rmse', 'normal_a1'], 
    multi_dataset_eval=False,
    exclude=[],
)

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=interval)
runner = dict(type='IterBasedRunner_AMP', max_iters=2000)

# optimizer
optimizer = dict(
     type='AdamW', 
     # encoder=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
     encoder=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-6),
     decoder=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
)

# schedule
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-6, by_epoch=False)

acc_batch = 1
batchsize_per_gpu = 4
thread_per_gpu = 1

Kubric_dataset=dict(
     data = dict(
          train=dict(
               pipeline=[dict(type='BGR2RGB'),
                    dict(type='LabelScaleCononical'),
                    dict(type='RandomResize',
                              prob=0.5,
                              ratio_range=(0.85, 1.15),
                              is_lidar=True
                         ),
                         dict(type='RandomCrop', 
                              crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                              crop_type='rand', 
                              ignore_label=-1, 
                              padding=[0, 0, 0]
                         ),
                         dict(type='RandomEdgeMask',
                              mask_maxsize=50, 
                              prob=0.2, 
                              rgb_invalid=[0,0,0], 
                              label_invalid=-1,
                         ),
                         dict(type='RandomHorizontalFlip', prob=0.4),
                         dict(type='PhotoMetricDistortion', 
                              to_gray_prob=0.1,
                              distortion_prob=0.1,
                         ),
                         dict(type='Weather',
                              prob=0.05),
                         dict(type='RandomBlur', 
                              prob=0.05),
                         dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                         dict(type='ToTensor'),
                         dict(type='Normalize', 
                              mean=[123.675, 116.28, 103.53], 
                              std=[58.395, 57.12, 57.375]
                         ),
               ],
               sample_size = -1,
          ),
          
          val=dict(
               pipeline=[dict(type='BGR2RGB'),
                         dict(type='LabelScaleCononical'),
                         dict(type='RandomCrop', 
                                   crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                                   crop_type='center', 
                                   ignore_label=-1, 
                                   padding=[0, 0, 0]),
                         dict(type='ToTensor'),
                         dict(type='Normalize', 
                              mean=[123.675, 116.28, 103.53], 
                              std=[58.395, 57.12, 57.375]
                         ),
               ],
               sample_size = -1,
          ),
     )
)