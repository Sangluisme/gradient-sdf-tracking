import torch
torch.set_default_dtype(torch.float64)

import torch.nn.functional as F
from third.camera_utils import rot_to_quat as Rot2Quat

def skew(w):
    return torch.Tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]],[-w[1], w[0], 0]]).type(torch.float64)

class SE3:

    dim = 4
    dof = 6

    def __init__(self, mat):

        assert mat.dim() == 2, "can only be 2D matrix"
        assert mat.size() == torch.Size([4, 4]), "can only be 4x4 torch tensor"
        # self.mat = mat
        self.mat = mat
        self.rotation = mat[:3,:3]
        self.translation = mat[:3,3]

    def update_left(self, mat):
        self.mat = mat @ self.mat
        self.rotation = self.mat[:3,:3]
        self.translation = self.mat[:3,3]


    def Quaternion(self):
        w = Rot2Quat(self.rotation)
        return torch.hstack([self.translation, w])
        
    @staticmethod
    def exp(Xi):

        Xi = Xi.type(torch.float64)
        
        assert Xi.size() == torch.Size([6]), "twist should have size 6"
        v = Xi[:3]
        w = Xi[-3:]

        w_skew = skew(w)
        norm_w = torch.norm(w, p=2).type(torch.float64)
        norm_w_inv = 1. / norm_w

        if(~torch.any(w)):
            R = torch.eye(3)
            t = v

        else:
            # A = torch.sin(norm_w) * norm_w_inv * torch.eye(3)
            # B = (norm_w_inv * norm_w_inv * w_skew) * (1-torch.cos(norm_w))
            # C = (1 - torch.sin(norm_w) * norm_w_inv) * norm_w_inv * norm_w_inv * w_skew  @ w_skew.T
            # R = A + B + C
            R = torch.eye(3) + norm_w_inv  * torch.sin(norm_w) * w_skew  + norm_w_inv * norm_w_inv * (1-torch.cos(norm_w)) * w_skew @ w_skew

            t = (torch.eye(3) + norm_w_inv * norm_w_inv * (1 - torch.cos(norm_w)) * w_skew + (norm_w - torch.sin(norm_w)) * norm_w_inv * norm_w_inv * norm_w_inv * w_skew @ w_skew) @ v

        
        T = torch.eye(4)

        T[:3,:3] = R
        T[:3, 3] = t

        return T

    @staticmethod
    def log(T):
        R = T[:3,:3]
        t = T[:3, 3]

        theta = torch.acos((torch.trace(R) - 1) / 2)
        Ksi = torch.zeros(6)

        if theta == 0:
            Ksi[-3:] = t 
        else:
            w = theta * (1 / (2*torch.sin(theta))) * torch.Tensor([[R[2,1] - R[1,2]], [R[0,2] - R[2,0]], [R[1,0] - R[0,1]]])
            wx = torch.Tensor([ [0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

            t = T

            theta_inv = 1 / theta
            A = (torch.eye(3) - 0.5 * wx) 
            b = 2 * (1 - torch.cos(theta))
            b_inv = 1 / b
            B = theta_inv * theta_inv * (1 - theta * torch.cos(theta)) * b_inv
            C = wx @ wx @ t

            v = A + B * C

            Ksi[:3] = v
            Ksi[-3:] = w

        return Ksi
    
        
