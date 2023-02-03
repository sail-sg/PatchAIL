Running commands:

Before run:

```
pip install opencv-python 
```

Hint: 
- `snapshot_epoch` means the model epoch to load (the suffix of the model pkl)
- `visualization_type` means the visualization type that you want to make


Examples on DMC task: cheetah_run:
```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=cheetah_run seed=1 +snapshot_epoch=1000000 +visualization_type=patch_rewards
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=cheetah_run seed=1 +snapshot_epoch=1000000 +visualization_type=feature_map
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=cheetah_run seed=1 +snapshot_epoch=1000000 +visualization_type=feature_weighted_patch_rewards
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=cheetah_run seed=1 +snapshot_epoch=1000000 +visualization_type=gradcam 
```

Examples on Atari task: pong
```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=atari obs_type=pixels suite/atari_task=pong seed=1 +snapshot_epoch=_final +visualization_type=patch_rewards
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=atari obs_type=pixels suite/atari_task=pong seed=1 +snapshot_epoch=_final +visualization_type=feature_map
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=atari obs_type=pixels suite/atari_task=pong seed=1 +snapshot_epoch=_final +visualization_type=feature_weighted_patch_rewards
```

```
CUDA_VISIBLE_DEVICES=0 python patch_visualize.py agent=patchirl suite=atari obs_type=pixels suite/atari_task=pong seed=1 +snapshot_epoch=_final +visualization_type=gradcam 
