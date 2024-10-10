'''
To Run this code, you must comment out 
# step physics and render each frame 
part since, this part is worked in pre_physics_step in
our code!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
from isaacgymenvs.cfg.terrain.terrain_cfg import TerrainCfg
from isaacgymenvs.utils.terrain import Terrain


class DyrosDynamicWalk(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length_s = self.cfg["env"]["episodeLength"]
        self.max_episode_length = self.max_episode_length_s / (self.cfg["sim"].get("dt") * self.cfg["env"].get("controlFrequencyInv", 8))
        self.num_obs_his = self.cfg["env"]["NumHis"]
        self.num_obs_skip = self.cfg["env"]["NumSkip"]
        self.initial_height = self.cfg["env"]["initialHieght"]

        self.num_single_step_obs = self.cfg["env"]["NumSingleStepObs"]
        self.num_action = self.cfg["env"]["NumAction"]

        self.cfg["env"]["numObservations"] = (self.num_single_step_obs+self.num_action)*(self.num_obs_his-1)+self.num_single_step_obs
        self.cfg["env"]["numActions"] = self.num_action
        self.perturb = self.cfg["env"]["perturbation"]

        self.terrain_cfg = TerrainCfg()
        
        self.init_done = False

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        #for PD controll
        self.Kp = torch.tensor([2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
            2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
            6000.0, 10000.0, 10000.0,
            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
            100.0, 100.0,
            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0], dtype=torch.float ,device=self.device) / 9.0

        self.Kv = torch.tensor([15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            200.0, 100.0, 100.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
            2.0, 2.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0], dtype=torch.float ,device=self.device) / 3.0

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # sensors_per_env = 2

        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, -1, 13)
        
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        self.initial_dof_pos[:,0:] = torch.tensor([0.0, 0.0, -0.24, 0.6, -0.36, 0.0, \
                                                    0.0, 0.0, -0.24, 0.6, -0.36, 0.0, \
                                                    0.0, 0.0, 0.0, \
                                                    0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,\
                                                    0.0, 0.0, \
                                                    -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0], device=self.device)
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.dof_pos[:] = self.initial_dof_pos[:]
        self.dof_vel[:] = self.initial_dof_vel[:]

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        #for Deep Mimic
        self.init_mocap_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
        self.mocap_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
        mocap_data_non_torch = np.genfromtxt('../assets/DeepMimic/processed_data_tocabi_walk.txt',encoding='ascii') 
        self.mocap_data = torch.tensor(mocap_data_non_torch,device=self.device, dtype=torch.float)
        self.mocap_data_num = int(self.mocap_data.shape[0] - 1)
        self.mocap_cycle_dt = 0.0005
        self.mocap_cycle_period = self.mocap_data_num * self.mocap_cycle_dt
        self.time = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"].get("dt")
        self.skipframe = self.cfg["env"].get("controlFrequencyInv", 8)
        self.dt_policy = self.dt*self.skipframe
        self.policy_freq_scale = 1/(self.dt_policy * 250) # e.g. 100/250
        self.sim_time_scale = self.dt / 0.0005 # e.g. 0.002 / 0.0005 (500Hz, 2000Hz)

        #for observation 
        self.qpos_noise = torch.zeros_like(self.dof_pos)
        self.qvel_noise = torch.zeros_like(self.dof_vel)
        self.qpos_pre = torch.zeros_like(self.dof_pos)
        #for random target velocity
        vel_mag = torch.rand(self.num_envs,1,device=self.device, dtype=torch.float, requires_grad=False)*0.8
        vel_theta = torch.rand(self.num_envs,1,device=self.device, dtype=torch.float, requires_grad=False)*0.0
        x_vel_target = vel_mag[:] * torch.cos(vel_theta[:])
        y_vel_target = vel_mag[:] * torch.sin(vel_theta[:])
        self.target_vel =  torch.cat([x_vel_target,y_vel_target],dim=1)

        
        #make motor scale constant
        self.motor_constant_scale = torch.rand(self.num_envs, 12, device=self.device, dtype=torch.float,requires_grad = False)*0.4+0.8
        #for normalizing observation
        obs_mean_non_torch = np.genfromtxt('../assets/Data/obs_mean_fixed.txt',encoding='ascii')
        obs_var_non_torch = np.genfromtxt('../assets/Data/obs_variance_fixed.txt',encoding='ascii')
        self.obs_mean = torch.tensor(obs_mean_non_torch,device=self.device,dtype=torch.float)
        self.obs_var = torch.tensor(obs_var_non_torch,device=self.device, dtype=torch.float)
        #initailize late update values
        self.pre_joint_velocity_states = self.dof_vel.clone()
        # self.pre_vec_sensor_tensor = self.vec_sensor_tensor.clone()
        self.action_torque_pre = torch.zeros(self.num_envs, 12, device = self.device, dtype=torch.float)
        self.contact_forces_pre = self.contact_forces.clone()

        # Bias
        self.qpos_bias = torch.rand(self.num_envs, 12, device=self.device, dtype=torch.float)*6.28/100-3.14/100
        self.quat_bias = torch.rand(self.num_envs, 3, device=self.device, dtype=torch.float)*6.28/150-3.14/150
        self.ft_bias = torch.rand(self.num_envs, 2, device=self.device, dtype=torch.float)*200.0-100.0
        self.m_bias = torch.rand(self.num_envs, 4, device=self.device, dtype=torch.float)*20.0-10.0

        #for running simulation
        self.action_torque = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
        self.target_data_qpos = torch.zeros_like(self.dof_pos, device = self.device, dtype = torch.float)
        self.target_data_force = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.float)
        self.delay_idx_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.simul_len_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.delay_idx_tensor[:,1] = 1
        self.simul_len_tensor[:,1] = 0
        for i in range(self.num_envs):
            self.delay_idx_tensor[i,0] = i
            self.simul_len_tensor[i,0] = i
        self.action_log = torch.zeros(self.num_envs, round(0.01/self.dt)+1, 12, device= self.device , dtype=torch.float)
        
        
        #for perturbation
        self.epi_len = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.epi_len_log = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.contact_reward_sum = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.contact_reward_mean = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.perturbation_count = torch.zeros(self.num_envs,device=self.device, dtype=torch.long)
        self.pert_duration = torch.randint(low=1, high=100, size=(self.num_envs,1),device=self.device, requires_grad=False).squeeze(-1)
        self.pert_on = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.impulse = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.magnitude = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.phase = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.perturb_timing = torch.ones(self.num_envs,device=self.device,dtype=torch.long,requires_grad=False)
        self.perturb_start = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        #target velocity change during episode
        self.vel_change_duration = torch.zeros(self.num_envs,device=self.device, dtype=torch.long)
        self.cur_vel_change_duration = torch.zeros(self.num_envs,device=self.device, dtype=torch.long) 
        self.start_target_vel = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.float)
        self.final_target_vel =  torch.zeros(self.num_envs,2,device=self.device,dtype=torch.float)    


        #modified observation
        self.actions = torch.zeros(self.num_envs, self.num_action, device=self.device, dtype=torch.float, requires_grad=False)
        self.actions_pre = torch.zeros(self.num_envs, self.num_action, device=self.device, dtype=torch.float, requires_grad=False)
        
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_his*self.num_obs_skip*self.num_single_step_obs, dtype=torch.float, requires_grad=False, device=self.device)
        self.action_history = torch.zeros(self.num_envs, self.num_obs_his*self.num_obs_skip*self.num_action, dtype=torch.float, requires_grad=False, device=self.device)

        self.init_done = True
       
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        mesh_type = self.terrain_cfg.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.total_mass = torch.zeros(self.num_envs, 1, device=self.device)
        for i in range(self.num_envs):
            robot_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.humanoid_handles[i])
            robot_masses = torch.tensor([prop.mass for prop in robot_props], dtype=torch.float, requires_grad=False, device=self.device)
            self.total_mass[i] = torch.sum(robot_masses)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_cfg.static_friction
        plane_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        plane_params.restitution = self.terrain_cfg.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain_cfg.horizontal_scale
        hf_params.row_scale = self.terrain_cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain_cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain_cfg.border_size 
        hf_params.transform.p.y = -self.terrain_cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.terrain_cfg.static_friction
        hf_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        hf_params.restitution = self.terrain_cfg.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain_cfg.border_size 
        tm_params.transform.p.y = -self.terrain_cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_cfg.static_friction
        tm_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        tm_params.restitution = self.terrain_cfg.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/dyros_tocabi/xml/dyros_tocabi.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # for getting motor gear infi
        self.action_high = torch.tensor([333, 232, 263, 289, 222, 166, \
                                        333, 232, 263, 289, 222, 166, \
                                        303, 303, 303, \
                                        64, 64, 64, 64, 23, 23, 10, 10,\
                                        10, 10, \
                                        64, 64, 64, 64, 23, 23, 10, 10], device=self.device)

        # create force sensors at the feet
        self.pelvis_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "base_link")
        self.right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Foot_Link")
        self.left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Foot_Link")
        # sensor_pose = gymapi.Transform(gymapi.Vec3(0, 0, -0.09))
        # sensor_props = gymapi.ForceSensorProperties()
        # sensor_props.enable_forward_dynamics_forces = True
        # sensor_props.enable_constraint_solver_forces = True
        # sensor_props.use_world_frame = False
        # self.gym.create_asset_force_sensor(humanoid_asset, self.left_foot_idx, sensor_pose, sensor_props)
        # self.gym.create_asset_force_sensor(humanoid_asset, self.right_foot_idx, sensor_pose, sensor_props)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        
        # create tensor of idxs of all the rigid bodies which isnt a feet
        self.non_feet_idxs = []
        for i in range(self.num_bodies):
            if (i != self.left_foot_idx and i!= self.right_foot_idx):
                    self.non_feet_idxs.append(i)

        self.initial_root_states = to_torch([0.0, 0.0, self.initial_height, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, requires_grad=False)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.initial_root_states[:3])

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self._get_env_origins()
        # env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        # env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        for i in range(self.num_envs):
            # create env instance
            # env_ptr = self.gym.create_env(
            #     self.sim, lower, upper, num_per_row
            # )
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            pos[2] +=  self.initial_height
            start_pose.p = gymapi.Vec3(*pos)
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.85938, 0.07813, 0.23438))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
            # dof_prop['friction'] = len(dof_prop['friction']) * [1.32]
            dof_prop['damping'] = len(dof_prop['damping']) * [0.1]
            dof_prop['armature'] = [0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                                    0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                                    0.078, 0.078, 0.078, \
                                    0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032, \
                                    0.0032, 0.0032, \
                                    0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032]
            dof_prop['velocity'] = len(dof_prop['velocity']) * [4.03]
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

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

    def compute_reward(self):
        self.rew_buf[:], reward, name, self.contact_reward_sum[:] = compute_humanoid_walk_reward(
            self.reset_buf,
            self.progress_buf,
            self.target_vel,
            self.root_states,
            self.target_data_qpos,
            self.target_data_force,
            self.dof_pos,
            self.initial_dof_vel,
            self.dof_vel,
            self.pre_joint_velocity_states,
            self.actions,
            self.actions_pre,
            # self.vec_sensor_tensor,
            # self.pre_vec_sensor_tensor,
            self.non_feet_idxs,
            self.contact_forces,
            self.contact_forces_pre,
            self.mocap_data_idx,
            self.termination_height,
            self.death_cost,
            self.policy_freq_scale,
            self.total_mass,
            self.contact_reward_sum,
            self.right_foot_idx,
            self.left_foot_idx
        )
        reward = torch.cat([reward, self.perturb_start], 1)

        if (self.terrain_cfg.curriculum and self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]):
            for i in range(self.terrain_cfg.num_cols):
                terrain_idx = (self.terrain_types == i).nonzero(as_tuple=False).squeeze(-1)
                terrain_level_mean = torch.sum(self.terrain_levels[terrain_idx]) / len(terrain_idx)
                reward = torch.cat([reward, terrain_level_mean*torch.ones_like(self.terrain_levels).unsqueeze(-1)], 1)

        name.append('perturbation')
        if (self.terrain_cfg.curriculum and self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]):
            for i in range(self.terrain_cfg.num_cols):
                name.append('terrain '+str(i)+' level')
        self.extras["stacked_rewards"] = reward
        self.extras["reward_names"] = name

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)   
        self.compute_humanoid_walk_observations()

    def start_perturbation(self, ids):
        self.pert_on[ids] = True
        self.impulse[ids] = torch.randint(low=50, high=250, size=(len(ids),1),device=self.device, requires_grad=False)
        self.pert_duration[ids] = torch.randint(low=int(0.1/self.dt_policy), high=int(1/self.dt_policy), size=(len(ids),1),device=self.device, requires_grad=False)#.squeeze(-1) # Unit: episode length
        self.magnitude[ids] = self.impulse[ids] / (self.pert_duration[ids] * self.dt_policy)
        self.phase[ids] = torch.rand(len(ids),1,device=self.device, dtype=torch.float, requires_grad=False)*2*3.14159265358979

    def finish_perturbation(self, ids):
        self.pert_on[ids] = False
        self.perturbation_count[ids] = 0

    def pre_physics_step(self, actions):   
        local_time = self.time % self.mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % self.mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + (local_time / self.mocap_cycle_dt).type(torch.long)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 

        mocap_data_idx_list = self.mocap_data_idx.squeeze(dim=-1)
        next_idx_list = next_idx.squeeze(dim=-1)

        self.target_data_qpos = cubic(local_time_plus_init, self.mocap_data[mocap_data_idx_list,0].unsqueeze(-1), self.mocap_data[next_idx_list,0].unsqueeze(-1), 
                                        self.mocap_data[mocap_data_idx_list,1:34], self.mocap_data[next_idx_list,1:34], 0.0, 0.0)
        self.target_data_force = cubic(local_time_plus_init, self.mocap_data[mocap_data_idx_list,0].unsqueeze(-1), self.mocap_data[next_idx_list,0].unsqueeze(-1), 
                                        self.mocap_data[mocap_data_idx_list,34:], self.mocap_data[next_idx_list,34:], 0.0, 0.0)

        self.actions = actions.to(self.device).clone()        
        positive_mask = self.actions[:,-1]>0
        self.actions[:,-1] = positive_mask * self.actions[:,-1] 
        self.action_history = torch.cat((self.action_history[:,self.num_actions:], self.actions),dim=-1)

        self.action_torque = self.actions[:,0:-1] * self.motor_constant_scale[:,0:]*self.action_high[:12]

        # new_vel_idx = torch.nonzero((self.epi_len[:] % (self.max_episode_length/4)) == (self.max_episode_length/4-1))
        # self.vel_change_duration[new_vel_idx] = torch.randint(low=1, high=int(1/self.dt_policy), size=(len(new_vel_idx),1),device=self.device, requires_grad=False)
        # self.cur_vel_change_duration[new_vel_idx] = 0  
        # self.start_target_vel[new_vel_idx] = self.target_vel[new_vel_idx].clone()
        # vel_mag = torch.rand(self.num_envs,1,device=self.device, dtype=torch.float, requires_grad=False) * 0.8
        # vel_theta = torch.rand(self.num_envs,1,device=self.device, dtype=torch.float, requires_grad=False)*0.0
        # x_vel_target = vel_mag[:] * torch.cos(vel_theta[:])
        # y_vel_target = vel_mag[:] * torch.sin(vel_theta[:])
        # self.final_target_vel[new_vel_idx] =  torch.cat([x_vel_target,y_vel_target],dim=1)[new_vel_idx] 

        # vel_change_idx = torch.nonzero(self.cur_vel_change_duration[:] < self.vel_change_duration[:])
        # self.cur_vel_change_duration[vel_change_idx] = self.cur_vel_change_duration[vel_change_idx] + 1
        # self.target_vel[:,0] = torch.where(self.cur_vel_change_duration[:] < self.vel_change_duration[:], \
        #                                   self.start_target_vel[:,0] + (self.final_target_vel[:,0]-self.start_target_vel[:,0]) * self.cur_vel_change_duration / self.vel_change_duration, \
        #                                   self.target_vel[:,0])
        # self.target_vel[:,1] = torch.where(self.cur_vel_change_duration[:] < self.vel_change_duration[:], \
        #                                   self.start_target_vel[:,1] + (self.final_target_vel[:,1]-self.start_target_vel[:,1]) * self.cur_vel_change_duration / self.vel_change_duration, \
        #                                   self.target_vel[:,1])
        
        if (self.perturb and (torch.mean(self.epi_len_log[:]) > self.max_episode_length - 8/self.dt_policy) and (torch.mean(self.contact_reward_mean[:]) > 0.165)):
            self.perturb_start[:, 0] = True
        # self.perturb_start[:, 0] = True
        if (self.perturb_start[0, 0] == True):
            forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            perturbation_start_idx = torch.nonzero((self.epi_len[:]%(8/self.dt_policy)==self.perturb_timing[:]))
            self.start_perturbation(perturbation_start_idx)
            self.perturbation_count = torch.where(self.pert_on,self.perturbation_count+1,self.perturbation_count)
            forces[:,self.pelvis_idx,0] = torch.where(self.pert_on,self.magnitude[:]*torch.cos(self.phase[:]),forces[:,-1,0])
            forces[:,self.pelvis_idx,1] = torch.where(self.pert_on,self.magnitude[:]*torch.sin(self.phase[:]),forces[:,-1,1])    
            perturbation_terminate_idx = (self.perturbation_count==self.pert_duration)
            self.finish_perturbation(perturbation_terminate_idx)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)          

        for _ in range(self.skipframe):
            mocap_torque = self.Kp*(self.target_data_qpos[:,:] - self.qpos_noise[:,:]) + self.Kv*(-self.dof_vel[:,:])
            upper_torque = self.Kp[12:]*(self.target_data_qpos[:,12:] - self.dof_pos[:,12:]) + self.Kv[12:]*(-self.dof_vel[:,12:])
            total_torque = torch.cat([self.action_torque,upper_torque], dim=1)
            stop_torque = self.Kp*(self.initial_dof_pos[:,:] - self.dof_pos[:,:]) + self.Kv*(-self.dof_vel[:,:])
            
            #action_log -> tensor(num_envs, time(current~past 9), dofs(33))
            self.action_log[:,0:-1,:] = self.action_log[:,1:,:] 
            self.action_log[:,-1,:] = self.action_torque
            self.simul_len_tensor[:,1] +=1
            self.simul_len_tensor[:,1] = self.simul_len_tensor[:,1].clamp(max=round(0.01/self.dt)+1, min=0)
            mask = self.simul_len_tensor[:,1] > self.delay_idx_tensor[:,1] 
            bigmask = torch.zeros(self.num_envs, 12,device=self.device, dtype=torch.bool)
            bigmask[:,:] = mask[:].unsqueeze(-1)
            delayed_lower_torque = torch.where(bigmask, self.action_log[self.delay_idx_tensor[:,0],self.delay_idx_tensor[:,1],:], \
                                        self.action_log[self.simul_len_tensor[:,0],-self.simul_len_tensor[:,1],:])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.cat([delayed_lower_torque,upper_torque], dim=1)))
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(total_torque))
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(stop_torque))
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(mocap_torque))
            
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            # self.qpos_noise = self.dof_pos + torch.clamp(0.00016/3.0*torch.randn_like(self.dof_pos) , min=-0.00016, max=0.00016)
            self.qpos_noise = self.dof_pos + torch.clamp(torch.normal(torch.zeros_like(self.dof_pos), 0.00016/3.0), min=-0.00016, max=0.00016)
            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.dt
            self.qpos_pre = self.qpos_noise.clone()

        self.epi_len[:] +=1
        

        self.render()
        #torch.cuda.empty_cache()
        #observation values update
        
        #time update
        self.time += self.dt_policy
        self.time += 5*self.dt_policy*self.actions[:,-1].unsqueeze(-1)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
            
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        #late update
        self.pre_joint_velocity_states = self.dof_vel.clone()
        self.action_torque_pre = self.action_torque.clone()
        self.contact_forces_pre = self.contact_forces.clone()
        self.actions_pre = self.actions.clone()


        # with open('force_data.txt', 'a') as f:
        #     # f.write('lfoot:')
        #     a = self.contact_forces[0,7,2].item()
        #     f.write(str(a))
        #     f.write('\t')
        #     a = self.contact_forces[0,14,2].item()
        #     f.write(str(a))
        #     f.write('\t')
        #     a = 9.81*self.vec_sensor_tensor[0,2].item()
        #     f.write(str(a))
        #     f.write('\t')
        #     a = 9.81*self.vec_sensor_tensor[0,8].item()
        #     f.write(str(a))
        #     f.write('\n')

    def check_termination(self):
        # reset agents
        # pelvis_height_env_idx  = self.root_states[:, 2] < (self.termination_height + self.env_origins[:,2])
        
        torso_rot = self.root_states[:,3:7].clone()
        identity_rot = torch.zeros_like(torso_rot)
        identity_rot[..., -1] = 1.
        quat_error = quat_diff_rad(identity_rot, torso_rot)
        orientation_env_idx = torch.abs(quat_error) > 0.5
        collision_true = torch.any(torch.norm(self.contact_forces[:, self.non_feet_idxs, :], dim=2) > 1., dim=1)

        reset = torch.where(orientation_env_idx, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        # reset = torch.where(pelvis_height_env_idx, torch.ones_like(self.reset_buf), reset)
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)

        self.reset_buf = torch.where(collision_true, torch.ones_like(self.reset_buf), reset)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if self.terrain_cfg.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        for i in env_ids:
            robot_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.humanoid_handles[i])
            robot_masses = torch.tensor([prop.mass for prop in robot_props], dtype=torch.float, requires_grad=False, device=self.device)
            self.total_mass[i] = torch.sum(robot_masses)

        self.qpos_noise[env_ids] = self.initial_dof_pos[env_ids].clone()
        self.qpos_pre[env_ids] = self.initial_dof_pos[env_ids].clone()
        self.qvel_noise[env_ids] = torch.zeros_like(self.qvel_noise[env_ids])

        self.qpos_bias[env_ids] = torch.rand(len(env_ids), 12, device=self.device, dtype=torch.float)*6.28/100-3.14/100
        self.quat_bias[env_ids] = torch.rand(len(env_ids), 3, device=self.device, dtype=torch.float)*6.28/150-3.14/150
        self.ft_bias[env_ids] = torch.rand(len(env_ids), 2, device=self.device, dtype=torch.float)*200.0-100.0
        self.m_bias = torch.rand(self.num_envs, 4, device=self.device, dtype=torch.float)*20.0-10.0

        self._reset_root_states(env_ids)
        self._reset_dof_states(env_ids)

        # reset target_vel & initial mocap_data (starting foot)
        vel_mag = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float, requires_grad=False) * 0.8
        vel_theta = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float, requires_grad=False)*0.0
        x_vel_target = vel_mag[:] * torch.cos(vel_theta[:])
        y_vel_target = vel_mag[:] * torch.sin(vel_theta[:])
        self.target_vel[env_ids] =  torch.cat([x_vel_target,y_vel_target],dim=1)
        
        rand = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float)
        mask = rand > 0.5
        self.init_mocap_data_idx[env_ids] = torch.where(mask, 0, 1800)
        
        #reset late update values
        self.pre_joint_velocity_states[env_ids] = self.initial_dof_vel[env_ids].clone()
        self.action_torque_pre[env_ids, : ] = 0
        # self.pre_vec_sensor_tensor[env_ids] = self.vec_sensor_tensor[env_ids].clone()
        self.contact_forces_pre[env_ids] = self.contact_forces[env_ids].clone()


        #reset time
        self.time[env_ids] = 0

        #reset motor constant scale
        self.motor_constant_scale[env_ids] = torch.rand(len(env_ids), 12, device=self.device, dtype=torch.float, requires_grad=False)*0.4+0.8


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.action_log[env_ids] = torch.zeros(1+round(0.01/self.dt),12,device=self.device,dtype=torch.float,requires_grad=False)
        self.delay_idx_tensor[env_ids,1] = torch.randint(low=1+int(0.002/self.dt),high=1+round(0.01 /self.dt),size=(len(env_ids),1),\
                                                        device=self.device,requires_grad=False).squeeze(-1)
        self.contact_reward_mean[env_ids] = self.contact_reward_sum[env_ids] /  self.epi_len[env_ids]
        self.contact_reward_sum[env_ids] = 0
        #low 5, high 12 for 2000 / 250Hz
        self.simul_len_tensor[env_ids,1] = 0
        self.epi_len_log[env_ids] = self.epi_len[env_ids]
        self.epi_len[env_ids] = 0


        #for perturbation
        self.perturbation_count[env_ids] = 0
        self.pert_on[env_ids] = False
        self.perturb_timing[env_ids] = torch.randint(low=0, high=int(8/self.dt_policy), size=(len(env_ids),1),device=self.device, requires_grad=False).squeeze(-1) 

        #for observation history buffer
        self.obs_history[env_ids,:] = 0
        self.action_history[env_ids,:] = 0

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.target_vel[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.terrain_cfg.max_init_terrain_level
            if not self.terrain_cfg.curriculum: max_init_level = self.terrain_cfg.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.terrain_cfg.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.terrain_cfg.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg["env"]['envSpacing']
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position

        if self.custom_origins:
            self.root_states[env_ids] = self.initial_root_states
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.initial_root_states
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_dof_states(self, env_ids):
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids], self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = 0.0  

        env_ids_int32 = env_ids.to(dtype=torch.int32)     
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_humanoid_walk_observations(self):
        pi =  3.14159265358979 
        q = self.root_states[:,3:7].clone()
        fixed_angle_x, fixed_angle_y, fixed_angle_z = quat2euler(q)

        fixed_angle_x += self.quat_bias[:,0]
        fixed_angle_y += self.quat_bias[:,1]
        fixed_angle_z += self.quat_bias[:,2]

        # lfoot_ft = self.contact_forces[:,self.left_foot_idx,:]
        # rfoot_ft = self.contact_forces[:,self.right_foot_idx,:]
        
        time2idx = (self.time % self.mocap_cycle_period) / self.mocap_cycle_dt
        phase = (self.init_mocap_data_idx + time2idx) % self.mocap_data_num / self.mocap_data_num
        sin_phase = torch.sin(2*pi*phase) 
        cos_phase = torch.cos(2*pi*phase)
        vel_noise = torch.rand(self.num_envs, 6, device=self.device, dtype=torch.float)*0.05-0.025
        obs = torch.cat((fixed_angle_x.unsqueeze(-1), fixed_angle_y.unsqueeze(-1), fixed_angle_z.unsqueeze(-1), 
                self.qpos_noise[:,0:12]+self.qpos_bias, 
                self.qvel_noise[:,0:12],
                sin_phase.view(-1,1),
                cos_phase.view(-1,1),
                self.target_vel[:,0].unsqueeze(-1),
                self.target_vel[:,1].unsqueeze(-1),
                self.root_states[:,7:]+vel_noise),dim=-1)
        
        diff = obs-self.obs_mean
        normed_obs =  diff/torch.sqrt(self.obs_var + 1e-8*torch.ones_like(self.obs_var))

        # for i in range(num_obs_skip*self.num_obs_his-1):
        #     obs_buffer[i] = obs_buffer[i+1]
        # obs_buffer[-1] = normed_obs.clone()
        
        self.obs_history = torch.cat((self.obs_history[:,self.num_single_step_obs:], normed_obs), dim=-1)

        epi_start_idx = (self.epi_len == 0)
        for i in range(self.num_obs_his*self.num_obs_skip):
            self.obs_history[epi_start_idx,self.num_single_step_obs*i:self.num_single_step_obs*(i+1)] = normed_obs[epi_start_idx,:]

        for i in range(0, self.num_obs_his):
            self.obs_buf[:,self.num_single_step_obs*i:self.num_single_step_obs*(i+1)] = \
                self.obs_history[:,self.num_single_step_obs*(self.num_obs_skip*(i+1)-1):self.num_single_step_obs*(self.num_obs_skip*(i+1))]
       
        action_start_idx = self.num_single_step_obs*self.num_obs_his
        for i in range(self.num_obs_his-1):
            self.obs_buf[:,action_start_idx+self.num_actions*i:action_start_idx+self.num_actions*(i+1)] = \
                self.action_history[:,self.num_actions*(self.num_obs_skip*(i+1)):self.num_actions*(self.num_obs_skip*(i+1)+1)]
                        
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_walk_reward(
    reset_buf,
    progress_buf,
    target_vel,
    root_pose_states,
    joint_position_target,
    force_target,
    joint_position_states,
    joint_velocity_init,
    joint_velocity_states,
    pre_joint_velocity_states,
    actions,
    actions_pre,
    non_feet_idxs,
    contact_forces,
    contact_forces_pre,
    mocap_data_idx,
    termination_height,
    death_cost,
    policy_freq_scale,
    total_mass,
    contact_reward_sum,
    right_foot_idx,
    left_foot_idx
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int], Tensor, Tensor, Tensor, float, float, float, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor, List[str], Tensor]
    
    #return angle difference between body(root link) quat & target quat (0,0,0,1)
    torso_rot = root_pose_states[:,3:7]
    identity_rot = torch.zeros_like(torso_rot)
    identity_rot[..., -1] = 1.
    quat_error = quat_diff_rad(identity_rot, torso_rot) #quat_error = normalize_angle(quat_error)
    mimic_body_orientation_reward = 0.3 * torch.exp(-13.2 * torch.abs(quat_error)) 
    #calculate joint position & velocity & regulate with target
    qpos_regulation = 0.35 * torch.exp(-2.0 * torch.norm((joint_position_target[:,0:] - joint_position_states[:,0:]), dim=1)**2)
    #calculate difference between initial q_vel, and q_vel now
    qvel_regulation = 0.05 * torch.exp(-0.01 * torch.norm((joint_velocity_init[:,0:] - joint_velocity_states[:,0:]), dim=1)**2)
    #penalize contact force & difference
    
    #1. Method (by sensor -> doesnt works well)
    
    # lfoot_force= 9.81*vec_sensor_tensor[...,0:3]
    # rfoot_force = 9.81*vec_sensor_tensor[...,6:9]
    # lfoot_force_pre = 9.81*pre_vec_sensor_tensor[...,0:3]
    # rfoot_force_pre = 9.81*pre_vec_sensor_tensor[...,6:9]

    #2, Method (by contact net force)
    lfoot_force = contact_forces[:,left_foot_idx,0:3]
    rfoot_force = contact_forces[:,right_foot_idx,0:3] 

    lfoot_force_pre = contact_forces_pre[:,left_foot_idx,0:3]
    rfoot_force_pre = contact_forces_pre[:,right_foot_idx,0:3]

    policy_freq_scale = 1
    # contact_force_penalty = 0.1 * torch.exp(-0.0005*(torch.norm(lfoot_force[:], dim=1) + torch.norm(rfoot_force[:], dim=1)))
    contact_force_diff_regulation = 0.2 * torch.exp(-0.01*policy_freq_scale*(torch.norm(lfoot_force[:]-lfoot_force_pre[:], dim=1) + \
                                                            torch.norm(rfoot_force[:]-rfoot_force_pre[:], dim=1)))
    #calculate torque input cost
    torque_regulation = 0.05 * torch.exp(-0.01 * torch.norm((actions[:,0:-1])*333,dim=1))
    #penalize difference of torque values
    torque_diff_regulation = 0.6 * torch.exp(-0.01*policy_freq_scale * torch.norm((actions[:,0:-1]-actions_pre[:,0:-1])*333, dim=1))
    #penalize difference of dof_velocities
    qacc_regulation = 0.05 * torch.exp(-20.0*torch.norm((joint_velocity_states[:,0:]-pre_joint_velocity_states[:,0:]), dim=1)**2)
    #track body velocity difference between target & state
    body_vel_reward = 0.3 * torch.exp(-3.0 * torch.norm((target_vel[:,0:] - root_pose_states[:,7:9]), dim=1)**2)
    #compare & track if foot contact phase synchronizes with refrence motion
    left_foot_contact = (lfoot_force[:,2].unsqueeze(-1) > 1.)
    right_foot_contact = (rfoot_force[:,2].unsqueeze(-1) > 1.)
  
    
    ones = torch.ones_like(body_vel_reward)
    zeros = torch.zeros_like(body_vel_reward)
    

    DSP = (3300 <= mocap_data_idx) & (mocap_data_idx < 3600) 
    DSP = DSP | (mocap_data_idx < 300) 
    DSP = DSP | ((1500 <= mocap_data_idx) & ( mocap_data_idx < 2100))
    RSSP = (300 <= mocap_data_idx) & (mocap_data_idx < 1500)
    LSSP = (2100 <= mocap_data_idx) & (mocap_data_idx < 3300)
    DSP_sync = DSP & right_foot_contact & left_foot_contact
    RSSP_sync = RSSP & right_foot_contact & ~left_foot_contact
    LSSP_sync = LSSP & ~right_foot_contact & left_foot_contact
    foot_contact_reward = torch.zeros_like(mimic_body_orientation_reward, dtype=torch.float)
    foot_contact_feeder = 0.2*torch.ones_like(foot_contact_reward, dtype=torch.float)
    foot_contact_reward = torch.where(DSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)
    foot_contact_reward = torch.where(RSSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)
    foot_contact_reward = torch.where(LSSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)

    contact_reward_sum += foot_contact_reward

    double_support_force_diff_regulation = torch.zeros_like(mimic_body_orientation_reward, dtype=torch.float)

    left_foot_thres = lfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*total_mass
    right_foot_thres = rfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*total_mass
    thres = left_foot_thres | right_foot_thres
    force_thres_penalty = torch.where(thres.squeeze(-1), -0.2*ones[:], zeros[:])

    contact_force_penalty_thres = 0.1*torch.exp(-0.007*(torch.norm(torch.clamp(lfoot_force[:,2].unsqueeze(-1) - 1.4*9.81*total_mass, min=0.0), dim=1) \
                                                            + torch.norm(torch.clamp(rfoot_force[:,2].unsqueeze(-1) - 1.4*9.81*total_mass, min=0.0), dim=1)))
    contact_force_penalty = torch.where(thres.squeeze(-1), contact_force_penalty_thres[:], 0.1*ones[:])
        
    left_foot_thres_diff = torch.abs(lfoot_force[:,2]-lfoot_force_pre[:,2]).unsqueeze(-1) > 0.2*9.81*total_mass/policy_freq_scale
    right_foot_thres_diff = torch.abs(rfoot_force[:,2]-rfoot_force_pre[:,2]).unsqueeze(-1) > 0.2*9.81*total_mass/policy_freq_scale
    thres_diff = left_foot_thres_diff | right_foot_thres_diff
    force_diff_thres_penalty = torch.where(thres_diff.squeeze(-1), -0.05*ones[:], zeros[:])    

    #Ignoring regulation terms
    # qacc_regulation *= 0
    # qvel_regulation *= 0
    # torque_regulation *= 0
    # torque_regulation *= 0
    # torque_diff_regulation *= 0
    # contact_force_diff_regulation *= 0
    # contact_force_penalty *= 0
    weight_scale = total_mass / 104.48
    force_ref_reward = 0.1*torch.exp(-0.001*(torch.abs(lfoot_force[:,2]+weight_scale.squeeze(-1)*force_target[:,0]))) +\
                        0.1*torch.exp(-0.001*(torch.abs(rfoot_force[:,2]+weight_scale.squeeze(-1)*force_target[:,1])))


    names = ["mimic_body_orientation_reward", "qpos_regulation", "qvel_regulation",\
        "contact_force_penalty", "torque_regulation", "torque_diff_regulation", "body_vel_reward",\
            "qacc_regulation", "foot_contact_reward", "contact_force_diff_regulation",\
                "double_support_force_diff_regulation","force_thres_penalty","force_diff_thres_penalty", "force_ref_reward"]
    
    reward = torch.stack([mimic_body_orientation_reward, qpos_regulation,qvel_regulation,\
        contact_force_penalty, torque_regulation, torque_diff_regulation, body_vel_reward,\
           qacc_regulation, foot_contact_reward, contact_force_diff_regulation,\
            double_support_force_diff_regulation, force_thres_penalty, force_diff_thres_penalty, force_ref_reward],1)

    total_reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + \
        torque_regulation + torque_diff_regulation + body_vel_reward + qacc_regulation + foot_contact_reward + \
        contact_force_diff_regulation + double_support_force_diff_regulation + force_thres_penalty + force_diff_thres_penalty + force_ref_reward

    #check if collision occured
    collision_true = torch.any(torch.norm(contact_forces[:, non_feet_idxs, :], dim=2) > 1., dim=1)

    # adjust reward for fallen agents
    # total_reward = torch.where(root_pose_states[:, 2] < termination_height, \
    #     torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(collision_true, torch.ones_like(total_reward) * death_cost, total_reward)    
    total_reward = torch.where(torch.abs(quat_error) > 0.5, torch.ones_like(total_reward) * death_cost, total_reward)

    reward = torch.where(collision_true.unsqueeze(-1), torch.ones_like(reward)* death_cost, reward)
    
    return total_reward, reward, names, contact_reward_sum