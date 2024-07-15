# DYROS TOCABI RL Locomotion Repository

This repository is dedicated to DYROS TOCABI RL Locomotion. It has been developed based on the NVIDIA IsaacGym's Preview4, which can be accessed here: [NVIDIA IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

For environment setup, execute `create_conda_env_rlgpu.sh`. This script creates a conda environment named `rlgpu4`. After activating the `rlgpu4` environment, run the following command in the `python` directory:
```bash
pip install -e .
```
Then, in the `python/IsaacGymEnvs` directory, execute the following:
```bash
pip install -e .
```

## Training Your First TOCABI Policy

To train your first TOCABI policy, execute the following command in the `python/IsaacGymEnvs/isaacgymenvs` directory:
```bash
python train.py task=DyrosDynamicWalk
```

## Inference with a Trained Model

To load a trained checkpoint and only perform inference (no training), pass `test=True` as an argument, along with the checkpoint name. To avoid rendering overhead, you may also want to run with fewer environments using `num_envs=64`:
```bash
python train.py task=DyrosDynamicWalk checkpoint=runs/DyrosDynamicWalk/nn/DyrosDynamicWalk.pth test=True num_envs=64 headless=False
```

## Expected Error

1. **ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory**
   - Solution: 
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tech/anaconda3/envs/rlgpu4/lib
   ```