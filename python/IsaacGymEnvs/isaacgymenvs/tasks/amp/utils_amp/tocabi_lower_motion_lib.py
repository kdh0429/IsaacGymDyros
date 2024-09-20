# Copyright (c) 2018-2023, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import yaml

from .motion_lib import MotionLib

from ..poselib.poselib.core.rotation3d import *

from isaacgymenvs.utils.torch_jit_utils import to_torch, slerp

class TocabiLowerMotionLib(MotionLib):
    def __init__(self, motion_file, num_dofs, device):
        self._num_dof = num_dofs - 21
        # self._key_body_ids = key_body_ids
        self._device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)
        print("Motion IDs: ", self.motion_ids)

        return

    # def sample_time(self, motion_ids, truncate_time=None):
    #     n = len(motion_ids)
    #     phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        
    #     motion_len = self._motion_lengths[motion_ids]
    #     if (truncate_time is not None):
    #         assert(truncate_time >= 0.0)
    #         motion_len -= truncate_time

    #     motion_time = phase * motion_len

    #     return motion_time


    def get_motion_state(self, motion_ids, motion_times):
        '''
        Get the motion state at the specified time
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
        0 : time
        1 ~ 12 : q pos
        13 ~ 24 : q vel
        25 ~ 27 : root pos
        28 ~ 31 : root rot
        32 ~ 34 : root vel
        35 ~ 37 : root ang vel
        38 ~ 43 : key pos
        44 ~ 51 : key rot
        '''
        n = len(motion_ids)
        # num_bodies = self._get_num_bodies()
        # num_key_bodies = self._key_body_ids.shape[0]

        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])
        root_rot1 = np.empty([n, 4])
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        # local_rot0 = np.empty([n, num_bodies, 4])
        # local_rot1 = np.empty([n, num_bodies, 4])
        dof_pos = np.empty([n, self._num_dof])
        dof_vel = np.empty([n, self._num_dof])
        key_pos0 = np.empty([n, 2, 3])
        key_pos1 = np.empty([n, 2, 3])
        
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, np.abs(dt))

        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :]  = curr_motion[frame_idx0[ids], 25:28]
            root_pos1[ids, :]  = curr_motion[frame_idx1[ids], 25:28]

            root_rot0[ids, :] = curr_motion[frame_idx0[ids], 28:32]
            root_rot1[ids, :]  = curr_motion[frame_idx1[ids], 28:32]

            root_vel[ids, :] = curr_motion[frame_idx0[ids], 32:35] * 0.0005 / dt[ids][:,np.newaxis]
            root_ang_vel[ids, :] = curr_motion[frame_idx0[ids], 35:38] * 0.0005 / dt[ids][:,np.newaxis]
            
            key_pos0[ids, 0, :] = curr_motion[frame_idx0[ids], 38:41] 
            key_pos0[ids, 1, :] = curr_motion[frame_idx0[ids], 41:44]

            key_pos1[ids, 0, :] = curr_motion[frame_idx1[ids], 38:41]
            key_pos1[ids, 1, :] = curr_motion[frame_idx1[ids], 41:44]

            dof_pos[ids, :] = curr_motion[frame_idx0[ids], 1:13]
            dof_vel[ids, :] = curr_motion[frame_idx0[ids], 13:25] * 0.0005 / dt[ids][:,np.newaxis]

        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)
        root_pos0 = to_torch(root_pos0, device=self._device)
        root_pos1 = to_torch(root_pos1, device=self._device)
        
        root_rot0 = to_torch(root_rot0, device=self._device)
        root_rot1 = to_torch(root_rot1, device=self._device)
        
        root_vel = to_torch(root_vel, device=self._device)
        root_ang_vel = to_torch(root_ang_vel, device=self._device)

        key_pos0 = to_torch(key_pos0, device=self._device)
        key_pos1 = to_torch(key_pos1, device=self._device)

        dof_pos = to_torch(dof_pos, device=self._device)
        dof_vel = to_torch(dof_vel, device=self._device)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_expand = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_expand) * key_pos0 + blend_expand * key_pos1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos
        # return dof_pos, dof_vel

    def _load_motions(self, motion_file):
        '''
        load motion files from txt
        0 : time
        1 ~ 12 : q pos
        13 ~ 24 : q vel
        25 ~ 27 : root pos
        28 ~ 31 : root rot
        32 ~ 34 : root vel
        35 ~ 37 : root ang vel
        38 ~ 43 : key pos
        '''
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        self._motion_classes = []

        total_len = 0.0

        motion_files, motion_weights, step_times, play_speeds = self._fetch_motion_files(motion_file)
        # motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = np.loadtxt(curr_file)
            
            if step_times[f] == 0.6:
                curr_motion = curr_motion[4400:9201, :]
            elif step_times[f] == 0.9:
                curr_motion = curr_motion[5600:12801, :]
            elif step_times[f] == "yaw":
                curr_motion = curr_motion[20000:54201, :]

            if play_speeds[f] is None:
                curr_dt = (curr_motion[1,0] - curr_motion[0,0]) 
                motion_fps = 1.0 / curr_dt

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)
                print("Motion length: {:.3f}s".format(curr_len))
            else:
                if play_speeds[f] < 0.0:
                    curr_motion = np.flip(curr_motion, axis=0)
                    play_speeds[f] = -play_speeds[f]

                curr_dt = (curr_motion[1,0] - curr_motion[0,0]) / play_speeds[f]
                motion_fps = np.abs(1.0 / curr_dt)

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)
                print("Motion length: {:.3f}s".format(curr_len))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        motion_files, motion_weights = super()._fetch_motion_files(motion_file)
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            step_times = []
            play_speeds = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                step_time = motion_entry.get('step_time', None)
                if (step_time is not None):
                    step_times.append(step_time)
                else:
                    step_times.append(None)
                
                play_speed = motion_entry.get('play_speed', None)
                if (play_speed is not None):
                    play_speeds.append(play_speed)
                else:
                    play_speeds.append(None)

        return motion_files, motion_weights, step_times, play_speeds

    # def _compute_motion_dof_vels(self, motion):
    #     num_frames = motion.shape[0]
    #     dt = motion[1,0] - motion[0,0]

    #     dof_vels = np.zeros([num_frames, self._num_dof])
    #     for f in range(num_frames - 1):
    #         dof_vels[f, :] = motion[f + 1, 1:34] - motion[f, 1:34]
        
    #     dof_vels[-1, :] = dof_vels[-2, :]

    #     return dof_vels