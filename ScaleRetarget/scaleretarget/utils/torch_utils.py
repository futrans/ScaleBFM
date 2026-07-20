import math
import numpy as np
import torch
import torch.nn.functional as F

# @torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    '''
    quat: scalar first
    '''

    shape = vec.shape
    
    quat = quat.reshape(-1,4)
    vec = vec.reshape(-1,3)
    
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2

    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

# @torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    '''
    q1,q2: scalar first
    '''
    
    shape = q1.shape

    q1 = q1.reshape(-1,4)
    q2 = q2.reshape(-1,4)

    w1,x1,y1,z1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    w2,x2,y2,z2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)