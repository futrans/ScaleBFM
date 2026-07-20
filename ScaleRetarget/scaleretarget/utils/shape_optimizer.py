import os
import gc
import torch
import smplx
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from rich.progress import Progress
from smplx.joint_names import JOINT_NAMES
from scipy.spatial.transform import Rotation as R
from hydra.core.hydra_config import HydraConfig

from scaleretarget.utils.kinematic_model.kinematics_model import KinematicsModel
from scaleretarget.utils.lafan_vendor.extract import read_bvh, read_bvh_multi_channel
from scaleretarget.utils.torch_utils import quat_apply, quat_mul


def _draw_shape_comparison(
    robot_body_3d,
    fitted_body_3d,
    fitted_label,
    output_name,
    drange,
):
    import matplotlib.pyplot as plt

    robot_body_3d = robot_body_3d - robot_body_3d[:, 0:1]
    fitted_body_3d = fitted_body_3d - fitted_body_3d[:, 0:1]
    output_path = os.path.join(HydraConfig.get().runtime.output_dir, output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(0, 45)
    ax.scatter(*robot_body_3d[0].T, label='Humanoid Robot Shape', c='blue')
    ax.scatter(*fitted_body_3d[0].T, label=fitted_label, c='red')
    ax.set_xlim(-drange, drange)
    ax.set_ylim(-drange, drange)
    ax.set_zlim(-drange, drange)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)

def optimize_smplx_shape(robot_type,
                         robot_xml_path, 
                         body_model_path,
                         optim_joint_matches, 
                         optim_iterations, 
                         device='cpu',
                         debug=False):
    
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )
        
    smplx_neutral_model = smplx.create(
        body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False
    ).to(kinematic_model_device)

    robot_dof_names = kinematic_model.dof_names
    robot_body_joint_names = kinematic_model.body_names
            
    smplx_body_rest_pose = np.zeros((1, 1 + smplx_neutral_model.NUM_BODY_JOINTS, 3), dtype=np.float32)
    smplx_body_joint_names = JOINT_NAMES

    match_config = OmegaConf.load(optim_joint_matches)
    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    smplx_body_joint_pick = [i[1] for i in joint_match_config] 
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    smplx_body_joint_pick_idx = [smplx_body_joint_names.index(j) for j in smplx_body_joint_pick]

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value

    smplx_pose_modifier = match_config.smplx_pose_modifier
    for mod_key, mod_value in smplx_pose_modifier.items(): # from smpl rest body pose to robot rest pose
        assert mod_key in smplx_body_joint_names, f"{mod_key} is not in SMPLX joint names!"
        smplx_body_rest_pose[:,smplx_body_joint_names.index(mod_key)] = R.from_euler("xyz", mod_value, degrees=True).as_rotvec()
    smplx_body_rest_pose = torch.from_numpy(smplx_body_rest_pose).float().to(kinematic_model_device).view(1, -1)
            
    smplx_output = smplx_neutral_model(
        betas=torch.zeros(1, 16, dtype=torch.float, device=kinematic_model_device), # (16,)
        global_orient=smplx_body_rest_pose[:, :3], # (N, 3)
        body_pose=smplx_body_rest_pose[:, 3:], # (N, 63)
        transl=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device), # (N, 3)
        left_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
        right_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
        jaw_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
        leye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
        reye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    joint_pos = smplx_output.joints # (1, num_joints, 3)
    root_trans_offset = joint_pos[:, 0]

    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=root_trans_offset,
        root_rot=torch.tensor([0,0,0,1],dtype=torch.float, device=kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    robot_smplx_shape = torch.zeros(1, 16, requires_grad=True, device=kinematic_model_device)
    shape_optimizer = torch.optim.Adam([robot_smplx_shape], lr=0.1)
            
    num_iterations = optim_iterations
    logger.info(f'[Loader] Optimizing the smplx shape parameters. It takes {num_iterations} iterations in total!')
            
    with Progress() as progress:
        task = progress.add_task("Iteration: 0 / Loss: NaN", total=num_iterations)
                
        for iter in range(num_iterations):
            joint_pos = smplx_neutral_model(
                betas=robot_smplx_shape, # (16,)
                global_orient=smplx_body_rest_pose[:, :3], # (N, 3)
                body_pose=smplx_body_rest_pose[:,3:], # (N, 63)
                transl=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device), # (N, 3)
                left_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
                right_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
                jaw_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                leye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                reye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                # expression=torch.zeros(num_frames, 10).float(),
                return_full_pose=True,
            ).joints

            diff = robot_body_pos[:, robot_body_joint_pick_idx].detach() - joint_pos[:, smplx_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum()

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
                    
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()

            progress.update(task, advance=1)
            
    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            joint_pos[:, smplx_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted SMPLX Shape',
            output_name=f"{robot_type}_optim_smplx_shape.png",
            drange=1,
        )

    shape = robot_smplx_shape.cpu().detach()
            
    gc.collect()

    return shape

def optimize_bvh_shape(robot_type,
                       robot_xml_path,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    """
    This only works for T-pose as rest pose; For example, LAFAN's rest pose is not T-pose so this function can not be applied.
    """
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,2]] = 0 # y up for BVH
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)
    rotation_matrix = torch.tensor([[1,0,0], [0,0,-1], [0,1,0]], dtype=torch.float32, device=kinematic_model_device)

    match_config = OmegaConf.load(optim_joint_matches)

    # bvh_root = match_config.bvh_root
    # bvh_root_index = bvh_data.bones.index(bvh_root)

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,-90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):
            rest_positions = torch.zeros((1, bvh_joint_num, 3), dtype=torch.float32, device=kinematic_model_device)
            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    rest_positions[:, i] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device).unsqueeze(0) * 100
                else:
                    rest_positions[:, i] = rest_positions[:, parent] + bvh_offset[i] * torch.exp(scale[i - 1])
            rest_positions = rest_positions @ rotation_matrix.T / 100
                
            diff = robot_body_pos[:, robot_body_joint_pick_idx] - rest_positions[:, bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")
            
    gc.collect()

    return scale

def optimize_actorcore_shape(robot_type,
                       robot_xml_path,
                       robot_root_height,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,robot_root_height], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,1]] = 0 # z up for converted BVH
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)

    match_config = OmegaConf.load(optim_joint_matches)

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,-90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):
            rest_positions = torch.zeros((1, bvh_joint_num, 3), dtype=torch.float32, device=kinematic_model_device)
            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    rest_positions[:, i] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device).unsqueeze(0)
                else:
                    rest_positions[:, i] = rest_positions[:, parent] + bvh_offset[i] * torch.exp(scale[i - 1])
            rest_positions = rest_positions / 100
                
            diff = robot_body_pos[:, robot_body_joint_pick_idx] - rest_positions[:, bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")
            
    gc.collect()

    return scale

def optimize_parc_shape(robot_type,
                       robot_xml_path,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = bvh_data.offsets
    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,1]] = 0
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)

    match_config = OmegaConf.load(optim_joint_matches)

    bvh_quats = np.zeros((bvh_joint_num,4),dtype=np.float32)
    bvh_quats[...,0] = 1 # scalar_first
    bvh_pose_modifier = match_config.bvh_pose_modifier
    for bvh_bone in bvh_pose_modifier:
        bvh_quats[bvh_joint_names.index(bvh_bone),:] = bvh_pose_modifier[bvh_bone]
    bvh_quats = torch.from_numpy(bvh_quats).float().to(kinematic_model_device)

    # global_state = quat_fk(bvh_quats, bvh_offset, bvh_data.parents) # weishuai: used for double check

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):

            global_positions = torch.zeros((1,bvh_joint_num,3),device=kinematic_model_device,dtype=torch.float)
            global_rotations = torch.zeros((1,bvh_joint_num, 4),device=kinematic_model_device,dtype=torch.float)
            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    global_positions[:,0] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device) * 100
                    # global_positions[0] = bvh_offset[0]
                    global_rotations[:,0] = bvh_quats[0]
                else:
                    parent_rot = global_rotations[:,parent]
                    local_offset = bvh_offset[i]
                    rotated_offset = quat_apply(parent_rot, local_offset)
                    
                    global_positions[:,i] = global_positions[:,parent] + rotated_offset * torch.exp(scale[i-1])
                    global_rotations[:,i] = quat_mul(parent_rot, bvh_quats[i])
            
            global_positions = global_positions / 100
                
            diff = robot_body_pos[:,robot_body_joint_pick_idx] - global_positions[:,bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            global_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")
            
    gc.collect()

    return scale

def optimize_motionmillion_shape(robot_type,
                       robot_xml_path,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,2]] = 0 # y up for BVH
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)
    rotation_matrix = torch.tensor([[1,0,0], [0,0,-1], [0,1,0]], dtype=torch.float32, device=kinematic_model_device)

    match_config = OmegaConf.load(optim_joint_matches)


    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,-90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):
            rest_positions = torch.zeros((1, bvh_joint_num, 3), dtype=torch.float32, device=kinematic_model_device)
            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    rest_positions[:, i] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device).unsqueeze(0)
                else:
                    rest_positions[:, i] = rest_positions[:, parent] + bvh_offset[i] * torch.exp(scale[i - 1])
            rest_positions = rest_positions @ rotation_matrix.T
                
            diff = robot_body_pos[:, robot_body_joint_pick_idx] - rest_positions[:, bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")
            
    gc.collect()

    return scale

def optimize_humoto_shape(robot_type,
                       robot_xml_path,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    """
    This only works for T-pose as rest pose; For example, LAFAN's rest pose is not T-pose so this function can not be applied.
    """
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,1]] = 0 # z up for BVH
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)

    match_config = OmegaConf.load(optim_joint_matches)

    # bvh_root = match_config.bvh_root
    # bvh_root_index = bvh_data.bones.index(bvh_root)

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,-90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):
            rest_positions = torch.zeros((1, bvh_joint_num, 3), dtype=torch.float32, device=kinematic_model_device)
            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    rest_positions[:, i] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device).unsqueeze(0) * 100
                else:
                    rest_positions[:, i] = rest_positions[:, parent] + bvh_offset[i] * torch.exp(scale[i - 1])
            rest_positions = rest_positions / 100
                
            diff = robot_body_pos[:, robot_body_joint_pick_idx] - rest_positions[:, bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")

    gc.collect()

    return scale

def optimize_geno_shape(robot_type,
                       robot_xml_path,
                       bvh_data,
                       optim_joint_matches,
                       optim_iterations,
                       device='cpu',
                       debug=False):
    """
    This only works for T-pose as rest pose; For example, LAFAN's rest pose is not T-pose so this function can not be applied.
    """
    kinematic_model_device = device

    kinematic_model = KinematicsModel(
        file_path=robot_xml_path,
        device=kinematic_model_device
    )

    robot_body_joint_names = kinematic_model.body_names
    robot_dof_names = kinematic_model.dof_names
    robot_root_pos = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device)

    bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
    bvh_offset[0, [0,2]] = 0 # y up for BVH
    bvh_joint_names = bvh_data.bones
    bvh_joint_num = len(bvh_joint_names)
    rotation_matrix = torch.tensor([[1,0,0], [0,0,-1], [0,1,0]], dtype=torch.float32, device=kinematic_model_device)

    match_config = OmegaConf.load(optim_joint_matches)

    bvh_quats = np.zeros((bvh_joint_num,4),dtype=np.float32)
    bvh_quats[...,0] = 1 # scalar_first
    bvh_pose_modifier = match_config.bvh_pose_modifier
    for bvh_bone in bvh_pose_modifier:
        bvh_quats[bvh_joint_names.index(bvh_bone),:] = bvh_pose_modifier[bvh_bone]
    bvh_quats = torch.from_numpy(bvh_quats).float().to(kinematic_model_device)

    robot_pose_modifier = match_config.robot_pose_modifier
    robot_body_rest_pose = torch.zeros(kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device)
    for mod_key, mod_value in robot_pose_modifier.items():
        assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
        robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
    robot_body_pos, _ = kinematic_model.forward_kinematics(
        root_pos=robot_root_pos.unsqueeze(0),
        root_rot=torch.from_numpy(R.from_euler("xyz", [0,0,-90], degrees=True).as_quat()).float().to(kinematic_model_device).unsqueeze(0),
        dof_pos=robot_body_rest_pose.unsqueeze(0)
    )

    joint_match_config = match_config.joint_matches
    robot_body_joint_pick = [i[0] for i in joint_match_config]
    bvh_body_joint_pick = [i[1] for i in joint_match_config]
    robot_body_joint_pick_idx = [robot_body_joint_names.index(j) for j in robot_body_joint_pick]
    bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]
    
    scale = torch.zeros((bvh_joint_num - 1,), dtype=torch.float, device=kinematic_model_device, requires_grad=True)
    scale_optimizer = torch.optim.Adam([scale], lr=0.01)
    
    num_iterations = optim_iterations
    logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")
    
    with Progress() as progress:
        task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

        for iter in range(num_iterations):

            global_positions = torch.zeros((1,bvh_joint_num,3),device=kinematic_model_device,dtype=torch.float)
            global_rotations = torch.zeros((1,bvh_joint_num, 4),device=kinematic_model_device,dtype=torch.float)

            for i in range(bvh_joint_num):
                parent = bvh_data.parents[i]
                if parent == -1:
                    assert i == 0
                    global_positions[:,0] = torch.tensor([0,0,0], dtype=torch.float, device=kinematic_model_device) * 100
                    global_rotations[:,0] = bvh_quats[0]
                else:
                    parent_rot = global_rotations[:,parent]
                    local_offset = bvh_offset[i]
                    rotated_offset = quat_apply(parent_rot, local_offset)
                    
                    global_positions[:,i] = global_positions[:,parent] + rotated_offset * torch.exp(scale[i-1])
                    global_rotations[:,i] = quat_mul(parent_rot, bvh_quats[i])
            
            rest_positions = global_positions @ rotation_matrix.T / 100

            diff = robot_body_pos[:, robot_body_joint_pick_idx] - rest_positions[:, bvh_body_joint_pick_idx]
            loss = diff.norm(dim=-1).square().sum() 

            progress.update(task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}")
            
            scale_optimizer.zero_grad()
            loss.backward()
            scale_optimizer.step()
        
            progress.update(task, advance=1)

    if debug:
        _draw_shape_comparison(
            robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy(),
            rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy(),
            fitted_label='Fitted BVH Shape',
            output_name=f"{robot_type}_optim_bvh_shape.png",
            drange=1.5,
        )

    scale = torch.exp(scale).cpu().detach().numpy()
    logger.info(f"[Loader] Optimized BVH scale: {scale}")
            
    gc.collect()

    return scale
