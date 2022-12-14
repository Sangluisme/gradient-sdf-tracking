import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from sdf_tracker.voxel_grid import *
from sdf_tracker.sdf import *
from sdf_tracker.volumetric_grad_sdf import *
from third.se3 import SE3 as SE3

def check_nan(array):
    if torch.isnan(array).any():
        print("there are nan in the array.")
        return True
    print("all good.")
    return False

class RigidPointOptimizer:

    def __init__(
        self,
        sdf,
        pose,
        device,
        num_iterations=50,
        conv_threshold=1e-5,
        damping=1.0):
        
        self.device = device
        self.num_iterations = num_iterations
        self.conv_threshold = torch.sqrt(torch.Tensor([conv_threshold]))
        self.conv_threshold_sq = conv_threshold
        self.damping = damping

        self.sdf = sdf
        self.pose = SE3(pose, device)


    def optimize(self, depth, K):
        
        h, w = depth.shape
        z = torch.from_numpy(depth.flatten()).to(self.device)
        valid = (z > self.sdf.z_min) & (z <= self.sdf.z_max)

        for iter in range(self.num_iterations):
            R = self.pose.rotation.to(self.device)
            t = self.pose.translation.to(self.device)


            # E = 0.0 # energy
            # g = torch.zeros(6)
            # H = torch.zeros(6,6)

            u, v = torch.meshgrid(torch.linspace(0,w-1, w), torch.linspace(0, h-1, h))
            x0 = u.T.flatten().to(self.device)
            y0 = v.T.flatten().to(self.device)

            x0 = (x0 - K[0,2]) / K[0,0]
            y0 = (y0 - K[1,2]) / K[1,1]

            points = torch.hstack([(x0*z).unsqueeze(-1), (y0*z).unsqueeze(-1), z.unsqueeze(-1)])

            points = points @ R.T + t

            w0 = self.sdf.weights(points)

            valid_w = w0 > 0

            phi0, grad, valid_d = self.sdf.tsdf(points)

            valid = (valid & valid_w & valid_d)
            # valid = (valid & valid_w)
            phi0 = torch.where(valid, phi0, torch.Tensor([0.0]).to(self.device))
            grad = grad * valid.unsqueeze(-1)
            
            E = torch.sum(phi0 * phi0)

            grad_xi = torch.zeros(6, points.shape[0]).to(self.device)

            grad_xi[:3, :] = grad.T
            grad_xi[-3:,:] = torch.cross(points.T, grad.T, dim=0)

            g =  grad_xi @ phi0
            H = grad_xi @ grad_xi.T

            counter = torch.sum(valid)

            E /= counter

            if(counter==0):
                print("zero points are valid.")
                return False


            xi = self.damping * torch.linalg.solve(H, g)

            #DEBUG:
            # print("---current xi: {0}\n".format(xi))

            if(torch.linalg.norm(xi)< self.conv_threshold.to(self.device)):
                print('------- convergence after {0} iterations'.format(iter))
                return True

            self.pose.update_left(self.pose.exp(-xi))

            #debug
            # print("---- pose: \n {0}".format(self.pose.mat))
            


        return False













