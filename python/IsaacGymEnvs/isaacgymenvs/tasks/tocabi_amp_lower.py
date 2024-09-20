# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from .amp.tocabi_amp_lower_base import TocabiAMPLowerBase
from .amp.utils_amp import gym_util
from .amp.utils_amp.tocabi_lower_motion_lib import TocabiLowerMotionLib

from isaacgymenvs.utils.torch_jit_utils import to_torch, calc_heading_quat_inv, torch_rand_float, my_quat_rotate
from isaacgym.torch_utils import quat2euler

NUM_AMP_OBS_PER_STEP = 1 + 3 + 12 + 12 + 6 # [root_h, root_rot, dof_pos, dof_vel, key_pos, key_quat]
# NUM_AMP_OBS_PER_STEP = 13 + 12 + 12 + 6 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel]


class TocabiAMPLower(TocabiAMPLowerBase):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3
        Custom = 4

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        state_init = cfg["env"]["stateInit"]
        self._state_init = TocabiAMPLower.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        motion_file = cfg['env'].get('motion_file', "tocabi_motion_data.txt")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/tocabi_motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        # Discriminator 에서 사용할 input (past, current) -> D(s, s')
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, False, key_pos)

        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat
        
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        self._motion_lib = TocabiLowerMotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                    #  key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == TocabiAMPLower.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == TocabiAMPLower.StateInit.Start
              or self._state_init == TocabiAMPLower.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == TocabiAMPLower.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        elif (self._state_init == TocabiAMPLower.StateInit.Custom):
            if self.custom_origins:
                self.update_terrain_level(env_ids)
                self._root_states[env_ids] = self._initial_root_states[env_ids]
                self._root_states[env_ids, :3] += self.env_origins[env_ids]
                self._root_states[env_ids, 3] += 0.97
                self._root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
                
                self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
                self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
            else:
                self._root_states[env_ids] = self._initial_root_states[env_ids]
                
                self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
                self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return
    
    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == TocabiAMPLower.StateInit.Random
            or self._state_init == TocabiAMPLower.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == TocabiAMPLower.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        dof_pos = torch.cat([dof_pos, self._initial_dof_pos[env_ids][:, 12:]], dim=-1)
        dof_vel = torch.cat([dof_vel, self._initial_dof_vel[env_ids][:, 12:]], dim=-1)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        '''
        비율에 따라 default 상태, 랜덤한 상태로 초기화
        '''
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        # print("motion time: ", motion_times[0])
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, False, key_pos)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, False, key_body_pos)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], False, key_body_pos[env_ids])
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, local_root_obs, key_pos):
    # type: (Tensor, Tensor, Tensor, bool, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    # root_rot_obs = root_rot
    # root_rot_obs = quat_to_tan_norm(root_rot_obs)

    fixed_angle_x, fixed_angle_y, fixed_angle_z = quat2euler(root_rot)
    root_rot_obs = torch.cat((fixed_angle_x.unsqueeze(-1), fixed_angle_y.unsqueeze(-1), fixed_angle_z.unsqueeze(-1)), dim=-1)

    # local_root_vel = my_quat_rotate(heading_rot, root_vel)
    # local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    if not local_root_obs:
        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_pos - root_pos_expand
    
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
        flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])
        local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    else:
        flat_local_key_pos = key_pos.view(-1, 6)
    
    obs = torch.cat((root_h, root_rot_obs, dof_pos[:,:12], dof_vel[:,:12], flat_local_key_pos), dim=-1)
    return obs