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



class TocabiNewWalk(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 30 #New Method:34
        self.cfg["env"]["numActions"] = 12
        
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
            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0], dtype=torch.float ,device=self.device)

        self.Kv = torch.tensor([15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            200.0, 100.0, 100.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
            2.0, 2.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0], dtype=torch.float ,device=self.device)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 2
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs,-1,13)
        self.head_states = self.rb_states[:,self.head_idx,:]
        self.lfoot_states = self.rb_states[:,self.left_foot_idx,:]
        self.rfoot_states = self.rb_states[:,self.right_foot_idx,:]

        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)        
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0
        
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


        #Loading processed data (only going to use upper body joint positions from here)
        self.init_mocap_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
        self.mocap_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
        mocap_data_non_torch = np.genfromtxt('../assets/DeepMimic/processed_data_tocabi_walk.txt',encoding='ascii') 
        self.mocap_data = torch.tensor(mocap_data_non_torch,device=self.device, dtype=torch.float)
        self.mocap_data_num = int(self.mocap_data.shape[0] - 1)
        self.mocap_cycle_dt = 0.0005
        self.time = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"].get("dt")

        #for observation 
        self.qpos_noise = torch.zeros_like(self.dof_pos)
        self.qvel_noise = torch.zeros_like(self.dof_vel)
        self.qvel_lpf = torch.zeros_like(self.dof_vel)
        self.qpos_pre = torch.zeros_like(self.dof_pos)
        #for random target velocity
        x_vel = torch.rand(self.num_envs, 1, device=self.device, dtype=torch.float)*0.7-0.2 #between 0.3~0.4
        y_vel = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.target_vel = torch.cat([x_vel,y_vel],dim=1)

        
        #for normalizing observation
        obs_mean_non_torch = np.genfromtxt('../assets/Data/obs_mean_fixed.txt',encoding='ascii')
        obs_var_non_torch = np.genfromtxt('../assets/Data/obs_variance_fixed.txt',encoding='ascii')
        self.obs_mean = torch.tensor(obs_mean_non_torch,device=self.device,dtype=torch.float)
        self.obs_var = torch.tensor(obs_var_non_torch,device=self.device, dtype=torch.float)
        


        #for running simulation

       
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

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
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # for getting motor gear infi
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.action_high = to_torch(motor_efforts, device=self.device)

        # create force sensors at the feet
        self.right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Foot_Link")
        self.left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Foot_Link")
        self.head_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "Head_Link")

        sensor_pose = gymapi.Transform()
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False
        self.gym.create_asset_force_sensor(humanoid_asset, self.right_foot_idx, sensor_pose, sensor_props)
        self.gym.create_asset_force_sensor(humanoid_asset, self.left_foot_idx, sensor_pose, sensor_props)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        
        # create tensor of idxs of all the rigid bodies which isnt a feet
        self.non_feet_idxs = []
        for i in range(self.num_bodies):
            if i != self.left_foot_idx:
                if i != self.right_foot_idx:
                    self.non_feet_idxs.append(i)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.92983, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.85938, 0.07813, 0.23438))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
            dof_prop['friction'] = len(dof_prop['friction']) * [0.1]
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
        self.rew_buf[:], self.reset_buf, reward, name = compute_humanoid_walk_reward(
            self.reset_buf,
            self.progress_buf,
            self.target_vel,
            self.root_states,
            self.dof_pos,
            self.dof_vel,
            self.vec_sensor_tensor,
            self.non_feet_idxs,
            self.contact_forces,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.initial_dof_pos,
            self.head_states,
            self.lfoot_states,
            self.rfoot_states,
            self.phase
        )
        self.extras["stacked_rewards"] = reward
        self.extras["reward_names"] = name

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)   
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.obs_buf[:], self.phase = compute_humanoid_walk_observations(
            self.root_states,
            self.qpos_noise,
            self.qvel_lpf,
            self.init_mocap_data_idx,
            self.time,
            self.mocap_cycle_dt,
            self.mocap_data_num,
            self.target_vel,
            self.obs_mean,
            self.obs_var
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids], self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = self.initial_dof_vel[env_ids]

        self.qpos_noise[env_ids] = self.initial_dof_pos[env_ids].clone()
        self.qpos_pre[env_ids] = self.initial_dof_pos[env_ids].clone()
        self.qvel_noise[env_ids] = torch.zeros_like(self.qvel_noise[env_ids])
        self.qvel_lpf[env_ids] = torch.zeros_like(self.qvel_lpf[env_ids])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset target_vel & initial mocap_data (starting foot)
        x = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float)
        self.target_vel[env_ids,0] = 0.7*x.squeeze()-0.2
        y = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float)
        mask = y > 0.5
        self.init_mocap_data_idx[env_ids] = torch.where(mask, 0, 1800)
        
        #reset time
        self.time[env_ids] = 0


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0



    def pre_physics_step(self, actions):

        self.actions = actions.to(self.device).clone()
        
        #### ignore#######
        mocap_cycle_period = self.mocap_data_num * self.mocap_cycle_dt
        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + 
                            (local_time / self.mocap_cycle_dt).type(torch.long)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 

        mocap_data_idx_list = self.mocap_data_idx.squeeze(dim=-1).tolist()
        next_idx_list = next_idx.squeeze(dim=-1).tolist()
        self.target_data_qpos = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx.tolist(),0], self.mocap_data[next_idx.tolist(),0], self.mocap_data[mocap_data_idx_list,1:34], self.mocap_data[next_idx_list,1:34], torch.zeros(self.num_envs,33, device = self.device, dtype = torch.float), torch.zeros(self.num_envs,33, device = self.device, dtype = torch.float))
        self.target_data_qpos = self.target_data_qpos.view(self.num_envs,self.num_dof)
        mocap_torque = self.Kp*(self.target_data_qpos[:,:] - self.qpos_noise[:,:]) + self.Kv*(-self.qvel_noise[:,:])
        ###############



        ones = torch.ones([self.num_envs,self.mocap_data.shape[1]-13],device=self.device) #reason for 13 : 1 for time 12 for actions
        self.qpos_input = torch.cat([self.actions, ones*self.mocap_data[0,13:]],dim=-1)
        PD_torque = self.Kp*(self.qpos_input[:,:] - self.qpos_noise[:,:]) + self.Kv*(-self.qvel_noise[:,:])

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(PD_torque))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(mocap_torque))
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.qpos_noise = self.dof_pos + torch.clamp( 0.00001/3.0*torch.randn_like(self.dof_pos) , min=-0.00001, max=0.00001)
        self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.dt
        self.qpos_pre = self.qpos_noise.clone()
        self.qvel_lpf = lpf(self.qvel_noise, self.qvel_lpf, 1/self.dt, 4.0) 



        self.render()
        #torch.cuda.empty_cache()
        #observation values update
        

        #time update
        self.time += self.dt
        
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()






#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_walk_reward(
    reset_buf,
    progress_buf,
    target_vel,
    root_pose_states,
    joint_position_states,
    joint_velocity_states,
    vec_sensor_tensor,
    non_feet_idxs,
    contact_forces,
    termination_height,
    death_cost,
    max_episode_length,
    q_nominal,
    head_states,
    lfoot_states,
    rfoot_states,
    phase
):
    # type: (Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor, List[int],Tensor,float,float,float,Tensor,Tensor,Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor, Tensor, List[str]]
    phase = phase.squeeze(-1)
    lfoot_norm = torch.abs(contact_forces[:,7,2])
    rfoot_norm = torch.abs(contact_forces[:,14,2])

    fly = (lfoot_norm[:]+rfoot_norm[:]==0)
    FalseTensor = torch.zeros_like(fly, dtype=torch.bool) 
    non_init = (0.04<phase) & (phase<0.5) | (phase>0.54)
    fly = torch.where(non_init, fly, FalseTensor)

    # lfoot_vel = torch.norm(lfoot_states[:,7:10],dim=1)
    # rfoot_vel =  torch.norm(rfoot_states[:,7:10],dim=1)

    lfoot_vel = lfoot_states[:,7]
    rfoot_vel = rfoot_states[:,7]


    ones = torch.ones_like(phase)
    zeros = torch.zeros_like(phase)
    grf_norm_l = scale_transform(saturate(lfoot_norm,zeros,100*9.81*0.5*ones),zeros,100*9.81*0.5*ones)
    grf_norm_r = scale_transform(saturate(rfoot_norm,zeros,100*9.81*0.5*ones),zeros,100*9.81*0.5*ones)
    vel_norm_l = scale_transform(saturate(lfoot_vel,zeros,0.3*ones),zeros,0.3*ones)
    vel_norm_r = scale_transform(saturate(rfoot_vel,zeros,0.3*ones),zeros,0.3*ones)

    sync_grf_l = sync_reward(phase)
    phase_plus_pi = torch.where(phase>0.5, phase-0.5, phase+0.5)
    sync_grf_r = sync_reward(phase_plus_pi)
    sync_vel_l = -sync_grf_l
    sync_vel_r = -sync_grf_r

    pi = 3.14159265358979
    grf_sync_reward = (torch.tan(pi/4*sync_grf_l*grf_norm_l)+torch.tan(pi/4*sync_grf_r*grf_norm_r))/2
    spd_sync_reward = (torch.tan(pi/4*sync_vel_l*vel_norm_l)+torch.tan(pi/4*sync_vel_r*vel_norm_r))/2
    root_vel_reward = torch.exp(-10*torch.norm((root_pose_states[:,7:9]-target_vel[:,:]),dim=1)**2)
    root_angvel_reward = torch.exp(-10*torch.norm(root_pose_states[:,10:],dim=1)**2)
    height_reward = torch.exp(-40*(root_pose_states[:,2]-1.0)**2)
    upper_reward = torch.exp(-10*torch.norm((root_pose_states[:,:2]-head_states[:,:2]),dim=1)**2)
    posture_reward = torch.exp(-torch.norm((joint_position_states[:,:]-q_nominal[:]),dim=1)**2)
    joint_vel_reward = torch.exp(-5e-6*torch.norm(joint_velocity_states[:,:],dim=1)**2) #Need To modify this!!!    

    # root_angvel_reward *= 0
    # joint_vel_reward *= 0
    # posture_reward *= 0 
    # height_reward *= 0
    # upper_reward *= 0

    print('grf')
    print(grf_sync_reward[19])
    print('spd')
    print(spd_sync_reward[19])
    print('phase')
    print(phase[19])
    print('foot_norm')
    print(rfoot_norm[19])
    print(lfoot_norm[19])
    names = ["grf_sync_reward", "spd_sync_reward", "root_vel_reward","root_angvel_reward","height_reward","upper_reward","posture_reward","joint_vel_reward"]
    
    reward = torch.stack([grf_sync_reward,spd_sync_reward,root_vel_reward,root_angvel_reward,height_reward,upper_reward,posture_reward,joint_vel_reward],1)

    total_reward = 0.225*grf_sync_reward + 0.225*spd_sync_reward + 0.1*root_vel_reward + 0.1*root_angvel_reward + 0.05*height_reward + 0.1*upper_reward + 0.1*posture_reward + 0.1*joint_vel_reward

    #Reset if leg converges
    leg_len = torch.norm(lfoot_states[:,:2] - rfoot_states[:,:2],dim=1)
    
    #check if collision occured
    collision_true = torch.any(torch.norm(contact_forces[:, non_feet_idxs, :], dim=2) > 1., dim=1)
    # adjust reward for fallen agents
    total_reward = torch.where(root_pose_states[:, 2] < termination_height, \
        torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(leg_len[:] < 0.1, \
        torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(collision_true, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(fly, torch.ones_like(total_reward) * death_cost, total_reward)

    reward = torch.where(root_pose_states[:,2].unsqueeze(-1) < termination_height, \
        torch.ones_like(reward)*death_cost, reward)
    reward = torch.where(leg_len[:].unsqueeze(-1) < 0.1, \
        torch.ones_like(reward)*death_cost, reward)
    reward = torch.where(collision_true.unsqueeze(-1), torch.ones_like(reward)* death_cost, reward)
    reward = torch.where(fly.unsqueeze(-1), torch.ones_like(reward)* death_cost, reward)
    
    # reset agents
    reset = torch.where(root_pose_states[:, 2] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(leg_len[:] < 0.1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    reset = torch.where(collision_true,torch.ones_like(reset_buf),reset)
    reset = torch.where(fly,torch.ones_like(reset_buf),reset)
    # reset = torch.where()
    return total_reward, reset, reward, names

# @torch.jit.script
# def compute_humanoid_walk_observations(
#     root_states,
#     qpos_noise,
#     qvel_lpf,
#     init_mocap_data_idx,
#     time,
#     mocap_cycle_dt,
#     mocap_data_num,
#     target_vel
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, int, Tensor) ->  Tuple[Tensor, Tensor]
#     pi =  3.14159265358979 
#     q = root_states[:,3:7]
#     r,p,y = get_euler_xyz(q)
#     root_ori = quat_from_euler_xyz(r,p,torch.zeros_like(y))
#     root_angvel = root_states[:,10:]
#     mocap_cycle_period = mocap_data_num * mocap_cycle_dt
#     time2idx = (time % mocap_cycle_period) / mocap_cycle_dt
#     phase = (init_mocap_data_idx + time2idx) % mocap_data_num / mocap_data_num
#     sin_phase = torch.sin(2*pi*phase) 
#     cos_phase = torch.cos(2*pi*phase)

#     print(root_ori[19])
#     print(q[19])
#     print('----------')
   
#     obs = torch.cat((root_ori, root_angvel, qpos_noise[:,0:12], 
#             qvel_lpf[:,0:12],
#             sin_phase.view(-1,1),
#             cos_phase.view(-1,1),
#             target_vel[:,0].unsqueeze(-1)), dim=-1)

#     return obs, phase


@torch.jit.script
def compute_humanoid_walk_observations(
    root_states,
    qpos_noise,
    qvel_lpf,
    init_mocap_data_idx,
    time,
    mocap_cycle_dt,
    mocap_data_num,
    target_vel,
    obs_mean,
    obs_var
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, int, Tensor, Tensor, Tensor) -> Tuple[Tensor,Tensor]
    pi =  3.14159265358979 
    q = root_states[:,3:7]
    fixed_angle_x,fixed_angle_y,fixed_angle_z = get_euler_xyz(q)
    mocap_cycle_period = mocap_data_num * mocap_cycle_dt
    time2idx = (time % mocap_cycle_period) / mocap_cycle_dt
    phase = (init_mocap_data_idx + time2idx) % mocap_data_num / mocap_data_num
    sin_phase = torch.sin(2*pi*phase) 
    cos_phase = torch.cos(2*pi*phase)
   
    obs = torch.cat((fixed_angle_x.unsqueeze(-1), fixed_angle_y.unsqueeze(-1), 
            fixed_angle_z.unsqueeze(-1), qpos_noise[:,0:12], 
            qvel_lpf[:,0:12],
            sin_phase.view(-1,1),
            cos_phase.view(-1,1),
            target_vel[:,0].unsqueeze(-1)), dim=-1)

    diff = obs-obs_mean
    return diff/torch.sqrt(obs_var + 1e-8*torch.ones_like(obs_var)), phase