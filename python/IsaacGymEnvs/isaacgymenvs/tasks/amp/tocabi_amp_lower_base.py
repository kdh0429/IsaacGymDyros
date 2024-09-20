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
import torch
from typing import Dict, Tuple, Any
import math

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, quat_diff_rad, torch_rand_float, quat_rotate_inverse
from isaacgym.torch_utils import quat2euler

from ..base.vec_task import VecTask

RED_BODY_IDS = [3,4,6,8, 11,12,14,16, 17,18, 19,22,24, 28, 29,32,34]
NUM_OBS = 3 + 6 + 3 + 12 + 12# [root_h, root_euler, root_vel, root_ang_vel, command, dof_pos, dof_vel]
NUM_ACTIONS = 12

KEY_BODY_NAMES = ["L_Foot_Link", "R_Foot_Link"]


class TocabiAMPLowerBase(VecTask):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.cfg = config
        self.custom_origins = False
        self.init_done = False

        self._pd_control = self.cfg["env"]["pdControl"]
        self.randomize = self.cfg["task"]["randomize"]
        self.noise =self.cfg["task"]["noise"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
       
        self.perturb = self.cfg["env"]["perturbation"]
        
        # command range
        self.c_x = self.cfg["env"]["command"]["x"]
        self.c_y = self.cfg["env"]["command"]["y"]
        self.c_yaw = self.cfg["env"]["command"]["yaw"]

        # obs history
        self.num_obs_his = self.cfg["env"]["NumHis"]
        self.num_obs_skip = self.cfg["env"]["NumSkip"]

        self.cfg["env"]["numObservations"] = (NUM_OBS + NUM_ACTIONS) * self.num_obs_his - NUM_ACTIONS
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        self.init_angle = torch.tensor([ 0.0, 0.0, -0.28, 0.6, -0.32, 0.0,
                        0.0, 0.0, -0.28, 0.6, -0.32, 0.0,
                        0.0, 0.0, 0.0,
                        0.3, 0.174533, 1.22173, -1.27, -1.57, 0.0, -1.0, 0.0,
                        0.0, 0.0,
                        -0.3, -0.174533, -1.22173, 1.27, 1.57, 0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)

        dt = self.cfg["sim"]["dt"]
        self.dt = dt #* self.control_freq_inv

        self.power_scale = torch.ones(self.num_envs, 12, device=self.device)
        
        # get gym GPU state tensors
        # 1) base state : xyz, quat, lin_vel, ang_vel
        # 2) dof state : dof pos, dof vel
        # 3) rigid body state : body pos, body rot, lin_vel, ang_vel
        # 4) net contact force : 로봇 body에 작용하는 net contact force

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        

        # FT sensor data for the feet  
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, 2 * 6)

        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # dof_pos, dof_vel 때문에 shape가 dof, 2로 되어 있음
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self._dof_vel_pre = self._dof_vel.clone()
        self.actions_pre = self.actions.clone()

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_pos[:] = self.init_angle
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.target_z = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.time_step = 0
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            for dof_name in control.p_gains.keys():
                if dof_name in self.dof_names[i]:
                    self.p_gains[i] = control.p_gains[dof_name] / 9.0
                    self.d_gains[i] = control.d_gains[dof_name] / 3.0

        # Bias & noises
        self.qpos_bias = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
        self.quat_bias = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.qpos_noise = torch.zeros_like(self._dof_pos)
        self.qvel_noise = torch.zeros_like(self._dof_vel)
        self.qpos_pre = torch.zeros_like(self._dof_pos)
        
        #for perturbation
        self.epi_len = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.epi_len_log = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.perturbation_count = torch.zeros(self.num_envs,device=self.device, dtype=torch.long)
        self.pert_duration = torch.randint(low=1, high=100, size=(self.num_envs,1),device=self.device, requires_grad=False).squeeze(-1)
        self.pert_on = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.impulse = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.magnitude = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.phase = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.perturb_timing = torch.ones(self.num_envs,device=self.device,dtype=torch.long,requires_grad=False)
        self.perturb_start = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        #target velocity change during episode
        self.vel_change = self.cfg["env"]["velChange"]
        self.vel_change_duration = torch.zeros(self.num_envs,device=self.device, dtype=torch.long)
        self.cur_vel_change_duration = torch.zeros(self.num_envs,device=self.device, dtype=torch.long) 
        self.start_target_vel = torch.zeros(self.num_envs,3,device=self.device,dtype=torch.float)
        self.final_target_vel =  torch.zeros(self.num_envs,3,device=self.device,dtype=torch.float) 

        # modified observation
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_his*self.num_obs_skip*NUM_OBS, dtype=torch.float, requires_grad=False, device=self.device)
        self.action_history = torch.zeros(self.num_envs, self.num_obs_his*self.num_obs_skip*self.num_actions, dtype=torch.float, requires_grad=False, device=self.device)
        
        # action delay
        self.delay_idx_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.simul_len_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.delay_idx_tensor[:,1] = 1
        self.simul_len_tensor[:,1] = 0
        for i in range(self.num_envs):
            self.delay_idx_tensor[i,0] = i
            self.simul_len_tensor[i,0] = i
        self.action_log = torch.zeros(self.num_envs, round(0.01/self.dt)+1, 12, device= self.device , dtype=torch.float)
        
        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True

        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        # Dynamic Randomization
        if self.randomize:
            self.power_scale[env_ids] = torch_rand_float(0.8, 1.2, (len(env_ids), 12), device=self.device)
            self.apply_randomizations(self.randomization_params)
        
        for i in env_ids:
            robot_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.humanoid_handles[i])
            robot_masses = torch.tensor([prop.mass for prop in robot_props], dtype=torch.float, requires_grad=False, device=self.device)
            self.total_mass[i] = torch.sum(robot_masses)
            
        # 로봇 위치 초기화
        self._reset_actors(env_ids)
        # sim 에서 데이터 업데이트
        self._refresh_sim_tensors()
        # sim에서 받은 데이터 기반으로 observation 계산
        self._compute_observations(env_ids)
        self._dof_vel_pre[env_ids,:] = 0.0
        self.actions_pre[env_ids,:] = 0.0

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self._root_states[env_ids] = self._initial_root_states[env_ids]
            self._root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self._root_states[env_ids] = self._initial_root_states[env_ids]

        self.commands_x[env_ids] = torch_rand_float(self.c_x[0], self.c_x[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.c_y[0], self.c_y[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.c_yaw[0], self.c_yaw[1], (len(env_ids), 1), device=self.device).squeeze()

        self.qpos_noise[env_ids] = self._initial_dof_pos[env_ids].clone()
        self.qpos_pre[env_ids] = self._initial_dof_pos[env_ids].clone()
        self.qvel_noise[env_ids] = torch.zeros_like(self.qvel_noise[env_ids])

        if self.noise:
            self.qpos_bias[env_ids] = torch.rand(len(env_ids), 12, device=self.device, dtype=torch.float)*6.28/100-3.14/100
            self.quat_bias[env_ids] = torch.rand(len(env_ids), 3, device=self.device, dtype=torch.float)*6.28/150-3.14/150
        else:
            self.qpos_bias[env_ids] = torch.zeros(len(env_ids), 12, device=self.device, dtype=torch.float)
            self.quat_bias[env_ids] = torch.zeros(len(env_ids), 3, device=self.device, dtype=torch.float)

        self.time_step = 0

        self.epi_len_log[env_ids] = self.epi_len[env_ids]
        self.epi_len[env_ids] = 0
        
        #for perturbation
        self.perturbation_count[env_ids] = 0
        self.pert_on[env_ids] = False
        self.perturb_timing[env_ids] = torch.randint(low=0, high=int(8/0.002), size=(len(env_ids),1),device=self.device, requires_grad=False).squeeze(-1) 

        #for observation history buffer
        self.obs_history[env_ids,:] = 0
        self.action_history[env_ids,:] = 0
        
        self.action_log[env_ids] = torch.zeros(1+round(0.01/self.dt),12,device=self.device,dtype=torch.float,requires_grad=False)
        self.delay_idx_tensor[env_ids,1] = torch.randint(low=1+int(0.002/self.dt),high=1+round(0.01 /self.dt),size=(len(env_ids),1),\
                                                        device=self.device,requires_grad=False).squeeze(-1)
        #low 5, high 12 for 2000 / 250Hz
        self.simul_len_tensor[env_ids,1] = 0
        return
    
    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(torch.tensor([0.2]))*6*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_envs(self, num_envs, spacing, num_per_row):
        # load humanoid asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        # asset options
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.upper_control_limit for prop in actuator_props]
        motor_efforts = motor_efforts[:12] 
        '''
        hasLimits - Flags whether the DOF has limits.
        lower - lower limit of DOF. In radians or meters
        upper - upper limit of DOF. In radians or meters
        driveMode - Drive mode for the DOF. See GymDofDriveMode.
        velocity - Maximum velocity of DOF. In Radians/s, or m/s
        effort - Maximum effort of DOF. in N or Nm.
        stiffness - DOF stiffness.
        damping - DOF damping.
        friction - DOF friction coefficient, a generalized friction force is calculated as DOF force multiplied by friction.
        armature - DOF armature, a value added to the diagonal of the joint-space inertia matrix. Larger values could improve simulation stability.
        '''
        
        # create force sensors at the feet
        self.pelvis_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "base_link")
        self.right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Foot_Link")
        self.left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Foot_Link")
        # sensor_pose = gymapi.Transform()
        # sensor_options = gymapi.ForceSensorProperties()
        # sensor_options.enable_forward_dynamics_forces = False # for example gravity
        # sensor_options.enable_constraint_solver_forces = True # for example contacts
        # sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        # self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose, sensor_options)
        # self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose, sensor_options)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.pelvis_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.dof_names = [self.gym.get_asset_dof_name(humanoid_asset, i) for i in range(self.num_dof)]

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.
        
        # 로봇 사이 간격 조정
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[2] += 0.97
                start_pose.p = gymapi.Vec3(*pos)    
            contact_filter = 0
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)


            for j in range(self.num_bodies):
                if j in RED_BODY_IDS:
                    self.gym.set_rigid_body_color(
                        env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.85938, 0.07813, 0.23438))
                else:
                    self.gym.set_rigid_body_color(
                        env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.20313, 0.20313, 0.20313))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
            # dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
            dof_prop["stiffness"].fill(0.0)
            dof_prop["damping"].fill(0.1)
            dof_prop["velocity"].fill(4.03)
            dof_prop['armature'] = [0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                0.078, 0.078, 0.078, \
                0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032, \
                0.0032, 0.0032, \
                0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032]
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            # self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle) #접촉 발 id

        robot_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        robot_masses = torch.tensor([prop.mass for prop in robot_props], dtype=torch.float, requires_grad=False, device=self.device)
        self.total_mass = torch.sum(robot_masses).repeat(self.num_envs,1)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = self.num_dof
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            curr_low = lim_low[j]
            curr_high = lim_high[j]
            curr_mid = 0.5 * (curr_high + curr_low)
            
            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = 1.0 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[j] = curr_low
            lim_high[j] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        return

    def _compute_reward(self, actions):
        self.rew_buf[:], reward_values, reward_names = compute_humanoid_reward(
            # self.obs_buf,
            self._root_states,
            # self.target_z,
            # self._initial_root_states,
            self._dof_vel,
            self._dof_vel_pre,
            self.commands,
            self.actions,
            self.actions_pre,
            self.motor_efforts,
            self._contact_forces,
            self.total_mass,
        )

        self.extras["reward_names"] = reward_names
        self.extras["reward_values"] = reward_values
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self._rigid_body_rot, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        '''
        sim에서 데이터를 받아와서 tensor로 변환
        '''
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        if (env_ids is None):
            ## 1. update obs history buffer  
            self.obs_history = torch.cat((self.obs_history[:,NUM_OBS:], obs), dim=-1)
            # ## 2. initialize obs buffer of new episodes 
            # epi_start_idx = (self.epi_len == 0)
            # for i in range(self.num_obs_his*self.num_obs_skip):
            #     self.obs_history[epi_start_idx,NUM_OBS*i:NUM_OBS*(i+1)] = obs
                
        else:
            ## 1. update obs history buffer  
            self.obs_history[env_ids] = torch.cat((self.obs_history[env_ids,NUM_OBS:], obs), dim=-1)
            ## 2. initialize obs buffer of new episodes 
            for i in range(self.num_obs_his*self.num_obs_skip):
                self.obs_history[env_ids,NUM_OBS*i:NUM_OBS*(i+1)] = obs

        ## 3. fill obs buffer with skipped obs history
        for i in range(0, self.num_obs_his):
            self.obs_buf[:,NUM_OBS*i:NUM_OBS*(i+1)] = \
                self.obs_history[:,NUM_OBS*(self.num_obs_skip*(i+1)-1):NUM_OBS*(self.num_obs_skip*(i+1))]
        ## 4. fill obs buffer with skipped action history       
        action_start_idx = NUM_OBS*self.num_obs_his
        for i in range(self.num_obs_his-1):
            self.obs_buf[:,action_start_idx+self.num_actions*i:action_start_idx+self.num_actions*(i+1)] = \
                self.action_history[:,self.num_actions*(self.num_obs_skip*(i+1)):self.num_actions*(self.num_obs_skip*(i+1)+1)]
        return
    
    def _compute_humanoid_obs(self, env_ids=None):
        self.time_step += 1
        if self.noise:
            basevel_noise = torch.rand(self.num_envs, 6, device=self.device, dtype=torch.float)*0.05-0.025
        else:
            basevel_noise = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)

        if (env_ids is None):
            root_states = self._root_states
            rootvel_noise = basevel_noise

            dof_pos = self.qpos_noise
            dof_pos_bias = self.qpos_bias
            quat_bias = self.quat_bias  
            dof_vel = self.qvel_noise


            # feet_sensors = self.vec_sensor_tensor
            key_pos = self._rigid_body_pos[:, self._key_body_ids, :]

            commands = self.commands

        else:
            root_states = self._root_states[env_ids]
            rootvel_noise = basevel_noise[env_ids]

            dof_pos = self.qpos_noise[env_ids]
            dof_pos_bias = self.qpos_bias[env_ids]
            quat_bias = self.quat_bias[env_ids]
            dof_vel = self.qvel_noise[env_ids]


            # feet_sensors = self.vec_sensor_tensor[env_ids]
            key_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

            commands = self.commands[env_ids]

        # obs = compute_humanoid_observations(root_states, dof_pos, dof_pos_bias, quat_bias, dof_vel, True, feet_sensors, key_pos, commands, target_z)
        obs = compute_humanoid_observations(root_states, rootvel_noise, dof_pos, dof_pos_bias, quat_bias, dof_vel, commands, key_pos)
                                 
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def start_perturbation(self, ids):
        self.pert_on[ids] = True
        self.impulse[ids] = torch.randint(low=50, high=250, size=(len(ids),1),device=self.device, requires_grad=False)
        self.pert_duration[ids] = torch.randint(low=int(0.1/0.002), high=int(1/0.002), size=(len(ids),1),device=self.device, requires_grad=False)#.squeeze(-1) # Unit: episode length
        # 외력 크기 및 방향 (2D)
        self.magnitude[ids] = self.impulse[ids] / (self.pert_duration[ids] * 0.002)
        self.phase[ids] = torch.rand(len(ids),1,device=self.device, dtype=torch.float, requires_grad=False)*2*3.14159265358979

    def finish_perturbation(self, ids):
        self.pert_on[ids] = False
        self.perturbation_count[ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        self.action_history = torch.cat((self.action_history[:,self.num_actions:], self.actions),dim=-1)

        if (self.perturb and (torch.mean(self.epi_len_log[:]) > self.max_episode_length - 8/0.002) ):
            self.perturb_start[:, 0] = True
        # self.perturb_start[:, 0] = True
        if (self.perturb_start[0, 0] == True):
            forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            perturbation_start_idx = torch.nonzero((self.epi_len[:]%(8/0.002)==self.perturb_timing[:]))
            self.start_perturbation(perturbation_start_idx)
            self.perturbation_count = torch.where(self.pert_on,self.perturbation_count+1,self.perturbation_count)
            forces[:,self.pelvis_idx,0] = torch.where(self.pert_on,self.magnitude[:]*torch.cos(self.phase[:]),forces[:,-1,0])
            forces[:,self.pelvis_idx,1] = torch.where(self.pert_on,self.magnitude[:]*torch.sin(self.phase[:]),forces[:,-1,1])    
            perturbation_terminate_idx = (self.perturbation_count==self.pert_duration)
            self.finish_perturbation(perturbation_terminate_idx)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)  

        if self.vel_change:
            vel_change = self.epi_len % int(self.max_episode_length / 2) == int(self.max_episode_length / 4 - 1)
            vel_change_idx = torch.nonzero(vel_change)
            # print(vel_change_idx)
            # if len(vel_change_idx) > 0:
                # if vel_change_idx[0] == 0:
                    # print("vel change")
            self.vel_change_duration[vel_change_idx] = torch.randint(low=1, high=250, size=(len(vel_change_idx),1),device=self.device, requires_grad=False)
            self.cur_vel_change_duration[vel_change_idx] = 0
            self.start_target_vel[vel_change_idx] = self.commands[vel_change_idx].clone()
            # if torch.mean(self.epi_len_log[:]) > self.max_episode_length - 10/0.002:
            #     self.final_target_vel[vel_change_idx,0] = torch_rand_float(0.8, 1.0, (len(vel_change_idx), 1), device=self.device)
            #     self.final_target_vel[vel_change_idx,1] = torch_rand_float(self.c_y[0], self.c_y[1], (len(vel_change_idx), 1), device=self.device)
            #     self.final_target_vel[vel_change_idx,2] = torch_rand_float(-0.2, 0.2, (len(vel_change_idx), 1), device=self.device)
            # else: 
            self.final_target_vel[vel_change_idx,0] = torch_rand_float(self.c_x[0], self.c_x[1], (len(vel_change_idx), 1), device=self.device)
            self.final_target_vel[vel_change_idx,1] = torch_rand_float(self.c_y[0], self.c_y[1], (len(vel_change_idx), 1), device=self.device)
            self.final_target_vel[vel_change_idx,2] = torch_rand_float(self.c_yaw[0], self.c_yaw[1], (len(vel_change_idx), 1), device=self.device)

            # if vel x is too high, set yaw to -pi/18 to pi/18
            # target_yaw_discout = torch_rand_float(-0.174, 0.174, (len(vel_change_idx), 1), device=self.device).view(-1)
            # self.final_target_vel[vel_change_idx,2] = torch.where(torch.logical_or(self.final_target_vel[vel_change_idx,0] > 0.8, self.final_target_vel[vel_change_idx,0] < -0.4), \
            #                                                         target_yaw_discout[vel_change_idx], self.final_target_vel[vel_change_idx,2])

            mask = self.cur_vel_change_duration < self.vel_change_duration
            for i in range(3):
                self.commands[:, i] = torch.where(mask, self.start_target_vel[:,i] + (self.final_target_vel[:,i] - self.start_target_vel[:,i]) * self.cur_vel_change_duration / self.vel_change_duration \
                                                                , self.commands[:,i])
            self.cur_vel_change_duration += mask.int()

        # print(self.commands[0,[0,2]])
        for _ in range(self.control_freq_inv):
            if (self._pd_control):
                pd_tar = self._action_to_pd_targets(self.actions)
                torque_lower = self.p_gains[:self.num_actions]*(pd_tar - self._dof_pos[:,:self.num_actions]) + \
                    self.d_gains[:self.num_actions]*(-self._dof_vel[:,:self.num_actions])   
                torques_upper = self.p_gains[self.num_actions:]*(self.init_angle[self.num_actions:] - self._dof_pos[:,self.num_actions:]) + \
                    self.d_gains[self.num_actions:]*(-self._dof_vel[:,self.num_actions:])
                torques_d = torch.cat((torque_lower, torques_upper), dim=1)
                force_tensor = gymtorch.unwrap_tensor(torques_d)
                self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
            else:
                torque_lower = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
                torque_lower = torch.clamp(torque_lower, -self.motor_efforts.unsqueeze(0), self.motor_efforts.unsqueeze(0))
                torques_upper = self.p_gains[self.num_actions:]*(self.init_angle[self.num_actions:] - self._dof_pos[:,self.num_actions:]) + \
                    self.d_gains[self.num_actions:]*(-self._dof_vel[:,self.num_actions:])
                
                #action_log -> tensor(num_envs, time(current~past 9), dofs(33))
                self.action_log[:,0:-1,:] = self.action_log[:,1:,:].clone() 
                self.action_log[:,-1,:] = torque_lower
                self.simul_len_tensor[:,1] +=1
                self.simul_len_tensor[:,1] = self.simul_len_tensor[:,1].clamp(max=round(0.01/self.dt)+1, min=0)
                mask = self.simul_len_tensor[:,1] > self.delay_idx_tensor[:,1] 
                bigmask = torch.zeros(self.num_envs, 12,device=self.device, dtype=torch.bool)
                bigmask[:,:] = mask[:].unsqueeze(-1)
                delayed_lower_torque = torch.where(bigmask, self.action_log[self.delay_idx_tensor[:,0],self.delay_idx_tensor[:,1],:], \
                                            self.action_log[self.simul_len_tensor[:,0],-self.simul_len_tensor[:,1],:])
                if self.noise:
                    torques_d = torch.cat((delayed_lower_torque, torques_upper), dim=1)
                else:
                    torques_d = torch.cat((torque_lower, torques_upper), dim=1)
                force_tensor = gymtorch.unwrap_tensor(torques_d)
                self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)


            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)

            if self.noise:
                self.qpos_noise = self._dof_pos + torch.clamp(torch.normal(torch.zeros_like(self._dof_pos), 0.00016/3.0), min=-0.00016, max=0.00016)
            else:
                self.qpos_noise = self._dof_pos

            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.dt
            self.qpos_pre = self.qpos_noise.clone()

        self.epi_len[:] +=1

            # self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(1.0,1.0,1.0), gymapi.Vec3(1.0,1.0,1.0), gymapi.Vec3(0.0,0.0,-1.0))
        self.render()
        return

    def post_physics_step(self):

        self.progress_buf += 1
        self.randomize_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        ###################################### print data section ######################################
        # Add root_state data to txt every loop
        # with open("/home/dyros/isacdata/curr_h.txt", "a") as file:
        #     root_state = self._root_states[0,2].cpu().numpy()
        #     file.write(f"{root_state}\n")
        # with open("/home/dyros/isacdata/vel_x.txt", "a") as file:
        #     root_state = self._root_states[0,7].cpu().numpy()
        #     file.write(f"{root_state}\n")
        # with open("/home/dyros/isacdata/tocabi_lower_torq_data.txt", "a") as file:
        #     action = self.actions[0, :12].cpu().numpy().tolist()
        #     action_str = '\t'.join(map(str, action))
        #     file.write(f"{action_str}\n")
        # with open("/home/dyros/isacdata/tocabi_lower_run_data.txt", "a") as file:
        #     root_state = self._root_states[0,:].cpu().numpy().tolist()
        #     q_ = self._dof_pos[0,:12].cpu().numpy().tolist()
        #     qdot_ = self._dof_vel[0,:12].cpu().numpy().tolist()
        #     key_pos = self._rigid_body_pos[0, self._key_body_ids, :].cpu().numpy().tolist()

        #     # Convert lists to strings with space-separated values
        #     root_state_str = '\t'.join(map(str, root_state))
        #     q_str = '\t'.join(map(str, q_))
        #     qdot_str = '\t'.join(map(str, qdot_))
        #     key_pos_str = '\t'.join('\t'.join(map(str, pos)) for pos in key_pos)
            
        #     # Write the formatted strings to the file
        #     file.write(f"{q_str}\t{qdot_str}\t{root_state_str}\t{key_pos_str}\n")
        
        # with open("/home/dyros/isacdata/tocabi_dof_torque.txt", "a") as file:
        #     torque_ = self.dof_force_tensor[0,:].cpu().numpy()
        #     torque_str = '\t'.join(map(str, torque_))    
        #     file.write(torque_str + '\n')

        # with open("/home/dyros/isacdata/tocabi_contact_force.txt", "a") as file:
        #     torque_sensorsL = self._contact_forces[0, self.left_foot_idx, :].cpu().numpy()
        #     torque_sensorsR = self._contact_forces[0, self.right_foot_idx, :].cpu().numpy()
        #     torque_sensors_strl = '\t'.join(map(str, torque_sensorsL))    
        #     torque_sensors_strr = '\t'.join(map(str, torque_sensorsR))    
        #     file.write(torque_sensors_strl + '\t' + torque_sensors_strr + '\n')
        # with open("/home/dyros/isacdata/tocabi_dof_vel.txt", "a") as file:
        #     qdot_ = self._dof_vel[0,:].cpu().numpy()
        #     qdot_str = '\t'.join(map(str, qdot_))    
        #     file.write(qdot_str + '\n')
        #################################################################################################
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        
        self._dof_vel_pre = self._dof_vel.clone()
        self.actions_pre = self.actions.clone()
        return
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def render(self):
        if self.viewer and self.camera_follow:
            print(self.commands[0,[0,2]])
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset[:self.num_actions] + self._pd_action_scale[:self.num_actions] * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
# def compute_humanoid_observations(root_states, dof_pos, dof_pos_bias, quat_bias, dof_vel, local_root_obs, feet_sensor, key_pos, commands, target_z):
def compute_humanoid_observations(root_states, rootvel_noise, dof_pos, dof_pos_bias, quat_bias, dof_vel, commands, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    '''
    제어기에 필요한 observation을 계산하는 함수
    1) root_h: root height (z)                  (1)
    2) root_rot: root rotation euler            (3)
    3) root_vel: root linear velocity           (3)
    4) root_ang_vel: root angular velocity      (3)
    5) command: command vel (x y yaw)           (3)
    6) dof_pos: dof position                    (12)
    7) dof_vel: dof velocity                    (12)
    '''
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10] + rootvel_noise[:, 0:3]
    root_ang_vel = root_states[:, 10:13] + rootvel_noise[:, 3:]

    root_h = root_pos[:, 2:3]

    fixed_angle_x, fixed_angle_y, fixed_angle_z = quat2euler(root_rot)
    fixed_angle_x += quat_bias[:, 0]
    fixed_angle_y += quat_bias[:, 1]
    fixed_angle_z += quat_bias[:, 2]

    local_root_vel = quat_rotate_inverse(root_rot, root_vel)
    # local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    
    dof_pos[:, :12] += dof_pos_bias

    ## key pos process
    # root_pos_expand = root_pos.unsqueeze(-2)
    # local_key_body_pos = key_pos - root_pos_expand

    # heading_rot_expand = heading_rot.unsqueeze(-2)
    # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
    #                                             heading_rot_expand.shape[2])
    # local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    obs = torch.cat((fixed_angle_x.unsqueeze(-1), fixed_angle_y.unsqueeze(-1), fixed_angle_z.unsqueeze(-1), local_root_vel, root_ang_vel, commands, dof_pos[:,:12], dof_vel[:,:12]), dim=-1)
    return obs

@torch.jit.script
# def compute_humanoid_reward(obs_buf, initial_root_states, root_states, dof_vel_pre, actions_pre, motor_efforts):
def compute_humanoid_reward(root_states, dof_vel, dof_vel_pre, commands, actions, actions_pre, motor_efforts, contact_force, total_mass):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, List[str]]
    '''
    Reward List
    - command vel x tracking
    
    - contact force threshold
    - contact force diff threshold

    - torque regulation
    - torque diff regulation
    - joint velocity regulation
    - joint acceleration regulation

    '''
    ## Reward Initialization
    reward = torch.zeros_like(root_states[:, 0])
    ############################### Tracking Command Rewards ###################################
    local_root_lin_vel = quat_rotate_inverse(root_states[:, 3:7], root_states[:, 7:10])

    _rew_lin_vel_x = 0.8 * torch.exp(-6.0 *torch.square(commands[:, 0] - local_root_lin_vel[:, 0]))
    _rew_lin_vel_y = 0.8 * torch.exp(-6.0 *torch.square(commands[:, 1] - local_root_lin_vel[:, 1])) 
    _rew_ang_vel_yaw = 0.6 * torch.exp(-7.0 *torch.square(commands[:, 2] - root_states[:, 12]))
    ############################### Contact Force Threshold Rewards ############################
    left_foot_threshold = contact_force[:, 8, 2].unsqueeze(-1) > 1.4*9.81*total_mass
    right_foot_threshold = contact_force[:, 16, 2].unsqueeze(-1) > 1.4*9.81*total_mass
    thres = left_foot_threshold | right_foot_threshold
    _reward_contact_force_threshold = -0.2 * torch.where(thres.squeeze(-1), torch.ones_like(root_states[:, 0]), torch.zeros_like(root_states[:, 0]))
    contact_force_penalty_thres = 0.1*(1-torch.exp(-0.007*(torch.norm(torch.clamp(contact_force[:, 8, 2].unsqueeze(-1) - 1.4*9.81*total_mass, min=0.0), dim=1) \
                                                         + torch.norm(torch.clamp(contact_force[:, 16, 2].unsqueeze(-1) - 1.4*9.81*total_mass, min=0.0), dim=1))))
    _contact_force_penalty = torch.where(thres.squeeze(-1), contact_force_penalty_thres[:], 0.1*torch.ones_like(root_states[:, 0]))

    ############################### Regulation Rewards #########################################
    _reward_joint_velocity_regulation = 0.05 * torch.exp(-0.01 * torch.norm((dof_vel[:,0:]), dim=1)**2)
    _reward_joint_acceleration_regulation = 0.05 * torch.exp(-20.0*torch.norm((dof_vel[:,0:]-dof_vel_pre[:,0:]), dim=1)**2)
    _reward_torque_regulation = 0.08 * torch.exp(-0.05 * torch.norm((actions[:,0:])*motor_efforts[:],dim=1))
    _reward_torque_diff_regulation = 0.6 * torch.exp(-0.01 * torch.norm((actions[:,0:]-actions_pre[:,0:])*motor_efforts[:], dim=1))
    
    
    reward += _rew_lin_vel_x
    # reward += _rew_lin_vel_y
    reward += _rew_ang_vel_yaw

    reward += (_reward_contact_force_threshold + _contact_force_penalty)

    reward += _reward_joint_velocity_regulation
    reward += _reward_joint_acceleration_regulation
    reward += _reward_torque_regulation
    reward += _reward_torque_diff_regulation

    reward_names = ['x_vel_tracking', 'y_vel_tracking','yaw_vel_tracking',\
                    'contact_force_threshold', 'contact_force_penalty', \
                    'joint_velocity_regulation', 'joint_acceleration_regulation', 'torque_regulation', 'torque_diff_regulation']
    reward_values = torch.stack([_rew_lin_vel_x, _rew_lin_vel_y, _rew_ang_vel_yaw, \
                                 _reward_contact_force_threshold, _contact_force_penalty, \
                                 _reward_joint_velocity_regulation, _reward_joint_acceleration_regulation, _reward_torque_regulation, _reward_torque_diff_regulation], dim=-1)

    return reward , reward_values, reward_names

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, rigid_body_rot,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone() 
        # 지지발은 제외
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 1.0, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, 1:] = False
        fly_height = body_height > 0.5
        fly_height[:, :8] = False
        fly_height[:, 9:16] = False
        fly_height[:, 17:] = False

        fall_height = torch.logical_or(fall_height, fly_height)
        
        fall_height = torch.any(fall_height, dim=-1)

        # has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.logical_or(fall_contact, fall_height)

        # quaternion error termination
        base_rot = rigid_body_rot[:, 0, :]
        
        init_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base_rot.device, dtype=base_rot.dtype).repeat(base_rot.shape[0], 1)
        base_rot = torch.abs(quat_diff_rad(init_rot, base_rot))
        base_terminate = base_rot > 3.141592 / 4.0

        has_fallen = torch.logical_or(has_fallen, base_terminate)        

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

class control:

    p_gains = {"L_HipYaw_Joint": 2000.0, "L_HipRoll_Joint": 5000.0, "L_HipPitch_Joint": 4000.0,
            "L_Knee_Joint": 3700.0, "L_AnklePitch_Joint": 3200.0, "L_AnkleRoll_Joint": 3200.0,
            "R_HipYaw_Joint": 2000.0, "R_HipRoll_Joint": 5000.0, "R_HipPitch_Joint": 4000.0,
            "R_Knee_Joint": 3700.0, "R_AnklePitch_Joint": 3200.0, "R_AnkleRoll_Joint": 3200.0,

            "Waist1_Joint": 6000.0, "Waist2_Joint": 10000.0, "Upperbody_Joint": 10000.0,

            "L_Shoulder1_Joint": 400.0, "L_Shoulder2_Joint": 1000.0, "L_Shoulder3_Joint": 400.0, "L_Armlink_Joint": 400.0,
            "L_Elbow_Joint": 400.0, "L_Forearm_Joint": 400.0, "L_Wrist1_Joint": 100.0, "L_Wrist2_Joint": 100.0,

            "Neck_Joint": 100.0, "Head_Joint": 100.0,            

            "R_Shoulder1_Joint": 400.0, "R_Shoulder2_Joint": 1000.0, "R_Shoulder3_Joint": 400.0, "R_Armlink_Joint": 400.0,
            "R_Elbow_Joint": 400.0, "R_Forearm_Joint": 400.0, "R_Wrist1_Joint": 100.0, "R_Wrist2_Joint": 100.0}

    d_gains = {"L_HipYaw_Joint": 15.0, "L_HipRoll_Joint": 50.0, "L_HipPitch_Joint": 20.0,
            "L_Knee_Joint": 25.0, "L_AnklePitch_Joint": 24.0, "L_AnkleRoll_Joint": 24.0,
            "R_HipYaw_Joint": 15.0, "R_HipRoll_Joint": 50.0, "R_HipPitch_Joint": 20.0,
            "R_Knee_Joint": 25.0, "R_AnklePitch_Joint": 24.0, "R_AnkleRoll_Joint": 24.0,

            "Waist1_Joint": 200.0, "Waist2_Joint": 100.0, "Upperbody_Joint": 100.0,

            "L_Shoulder1_Joint": 10.0, "L_Shoulder2_Joint": 28.0, "L_Shoulder3_Joint": 10.0, "L_Armlink_Joint": 10.0,
            "L_Elbow_Joint": 10.0, "L_Forearm_Joint": 10.0, "L_Wrist1_Joint": 3.0, "L_Wrist2_Joint": 3.0,

            "Neck_Joint": 100.0, "Head_Joint": 100.0,            

            "R_Shoulder1_Joint": 10.0, "R_Shoulder2_Joint": 28.0, "R_Shoulder3_Joint": 10.0, "R_Armlink_Joint": 10.0,
            "R_Elbow_Joint": 10.0, "R_Forearm_Joint": 10.0, "R_Wrist1_Joint": 3.0, "R_Wrist2_Joint": 3.0}
    
# terrain generator
from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
