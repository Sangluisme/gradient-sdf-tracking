import torch
import torch.nn.functional as F
import cv2 as cv
import sys

from sdf_tracker.voxel_grid import *
from sdf_tracker.sdf import *
from third.normal_estimator import *
import third.general as utils
from third.meshing import *

def check_nan(array):
    if torch.isnan(array).any():
        print("there are nan in the array.")
        return True
    print("all good.")
    return False

# torch.set_default_dtype(torch.float64)

class VolumetricGradSdf(VoxelGrid, Sdf):
    torch.set_default_dtype(torch.float64)

    def __init__(
        self,
        normal_estimator,
        T,
        device,
        grid_dim,
        z_min=0.5,
        z_max=3.5,
        voxel_size=0.02,
        shift=torch.Tensor([[0.0, 0.0, 0.0]]),
        counter=0
    ):
        VoxelGrid.__init__(self, grid_dim, device, voxel_size, shift)

        Sdf.__init__(self, T=T*voxel_size, device=device, z_min=z_min, z_max=z_max, counter=counter)

        self.Nest = normal_estimator
        self.device = device
        # self.tsdf_ = []
        self.vis = torch.ones(self.num_voxels, 1).type(torch.int32).to(self.device)
       

        self.dist = (-torch.ones(self.num_voxels) * self.T).to(self.device)
        self.grad = torch.zeros_like(self.grid_points).to(self.device)
        self.color = torch.zeros_like(self.grid_points).to(self.device)
        self.w = torch.zeros(self.num_voxels).to(self.device)
        
            
    # compute the new dist 
    def tsdf(self, points):
        index = self.nearest_index(points)
        valid = (index > 0)

        grad = torch.zeros_like(points)
        dist = torch.zeros(points.shape[0])

        disp = self.voxel2world(self.world2voxel(points)) - points

        grad = self.grad[index,:] * valid.unsqueeze(-1).type(torch.float32)
        dist = self.dist[index]

        normal = F.normalize(grad, p=2, dim=1)
     
        tsdf =  dist + torch.sum(normal * disp, dim=1)

        tsdf = torch.where(valid, tsdf, torch.Tensor([self.T]).to(self.device))

        return tsdf, normal, valid

    # compute the new weight
    def weights(self, points):
        index = self.nearest_index(points)
        valid = (index > 0)
        w = self.w[index]
        
        w = torch.where(valid, w, torch.Tensor([0.0]).to(self.device))

        return w
    
    def getVoxel(self, index):
        return self.tsdf_[self.idx2line(index)]


    def compute_centroid(self, K, depth):
        h, w = depth.shape
        u, v = torch.meshgrid(torch.linspace(0,w-1, w), torch.linspace(0, h-1, h))
        u = u.T.flatten().to(self.device)
        v = v.T.flatten().to(self.device)
        x0 = (u - K[0,2]) / K[0,0]
        y0 = (v - K[1,2]) / K[1,1]

        z = torch.from_numpy(depth.flatten()).to(self.device)

        valid = (z > self.z_min) & (z< self.z_max)
        z = torch.where(valid, z, torch.Tensor([0.0]).to(self.device))
        points = torch.vstack([x0*z, y0*z, z]).T

        counter = torch.sum(valid)

        centroid = torch.sum(points, dim=0) / counter

        self.shift = centroid # 0.0248 0.0205 3.69
        self.origin = self.shift - (0.5 * self.voxel_size * torch.Tensor(self.grid_dim).type(torch.float32)).to(self.device)
        print("...computed the centroid of first frame: {0} ".format(centroid))

        return centroid


    def update(self, rgb, depth, K, pose):

        med_depth = cv.medianBlur(depth, 5)

        Nest = self.Nest.compute(depth=med_depth)

        grad = Nest['normal']
        n_sq_inv = Nest['n_sq_inv']

        R = pose[:3,:3].to(self.device)
        t = pose[:3,3].to(self.device)

        points = self.voxel2world(self.grid_points)
        points = (points - t.unsqueeze(-2)) @ R

        project_points = points @ torch.from_numpy(K.T).type(torch.float32).to(self.device)
        n = torch.round(project_points[:,0]/project_points[:,2]).type(torch.int64)
        m = torch.round(project_points[:,1]/project_points[:,2]).type(torch.int64)

        valid = (n>=0) & (m>=0) & (n<depth.shape[1]) & (m<depth.shape[0])

        nn = torch.where(valid, n, -1).cpu()
        mm = torch.where(valid, m, -1).cpu()

        z = med_depth[mm,nn]      
        color = torch.from_numpy(rgb[mm,nn,:]).to(self.device)

        valid_z = torch.from_numpy((z>self.z_min) & (z < self.z_max)).to(self.device)

        # Debug: 
        sdf = torch.from_numpy(z).to(self.device) - points[:,2]
        w = self.weight(sdf)

        # trucate sdf
        sdf = self.truncate(sdf)
        valid_weight = (w>0)

        # normal
        normal = torch.from_numpy(grad[:,mm,nn]).type(torch.float32).to(self.device)
        n_sq_inv_points = torch.from_numpy(n_sq_inv[mm,nn]).to(self.device)
        

        # normal norm smaller than 0.1 is invalide
        valid_n = torch.norm(normal, p=2, dim=0)>=0.1
        xy_hom =  points * (1/points[:,2]).view(points.shape[0], 1)

        # normal angle is less then 75 degree
        valid_angle = torch.sum(normal.T*xy_hom,dim=1)*torch.sum(normal.T*xy_hom,dim=1)*n_sq_inv_points > (0.15*0.15)

        valid = (valid  & valid_weight & valid_z & valid_n  & valid_angle)

        # Debug: 
        print("-----DEBUG: valid grid number: ", torch.sum(valid))

        # prepare for update
        w = torch.where(valid, w, torch.Tensor([0.]).to(self.device))
        
        # update voxel properties
        self.w += w
        self.dist = torch.where(valid, self.dist + (sdf - self.dist) * w / self.w, self.dist)
        self.grad = torch.where(valid.unsqueeze(-1), self.grad- (R @ normal).T * w.unsqueeze(-1), self.grad)
        self.color = torch.where(valid.unsqueeze(-1), self.color + (color - self.color) * w.unsqueeze(-1) / self.w.unsqueeze(-1), self.color)
        
        # #debug
        # if check_nan(self.grad):
        #     print("there is nan in self grad.")
        
        if self.counter:
            vis = torch.ones(self.num_voxels, 1).type(torch.int32).to(self.device)
            vis[:,0] = torch.where(valid, vis[:,0], torch.Tensor([0]).type(torch.int32).to(self.device))
            self.vis = torch.hstack([self.vis, vis])
        else:
            self.vis[:,0]= torch.where(valid, self.vis[:,0], torch.Tensor([0]).type(torch.int32).to(self.device))


        print('-----counter: ', self.counter)
        print('-----DEBUG: visbility length: ', self.vis.shape[1])

        
    def setup(self, rgb, depth, K):
        pose = torch.eye(4)
        self.update(rgb, depth, K, pose)


    def export_pc(self, filename):
        normal = F.normalize(self.grad, p=2, dim=1)
        points = self.voxel2world(self.grid_points) - self.dist.view(self.num_voxels, 1) * normal
        colors = (self.color * 255).type(torch.int32)
        valid = (self.w > 0.0) & (torch.abs(self.dist) < torch.sqrt(torch.Tensor([3.0])).to(self.device)*self.voxel_size)
        count = torch.sum(valid)
        
        normal = normal[valid.nonzero(),:].cpu().squeeze()
        points = points[valid.nonzero(),:].cpu().squeeze()
        colors = colors[valid.nonzero(),:].cpu().squeeze()
        
        verts = np.hstack((points, normal, colors))
        
        verts_tuple = np.zeros(
        (count, ),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),('nx', 'f4'),('ny', 'f4'), ('nz', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')]
        )

        for i in range(0, count):
            verts_tuple[i] = tuple(verts[i, :])
           

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        # el_normal = plyfile.PlyElement.describe(vertex, "vertex")
        ply_data = plyfile.PlyData([el_verts])
        
        ply_data.write(filename)
        
    def export_mesh(self, filename):
        # x_min, x_max, y_min, y_max, z_min, z_max = boundary
        
        volumn_= self.dist.reshape((self.grid_dim[0], self.grid_dim[1], self.grid_dim[2])).permute(2,1,0)
        # volumn = volumn_[x_min:x_max, y_min:y_max, z_min:z_max]
    
        # utils.export_mesh(volumn, filename)
        verts, faces, normals, values = convert_sdf_samples_to_ply(
                                        volumn_.cpu(),
                                        self.origin.cpu(),
                                        self.voxel_size
                                        )
        
        save_ply(verts, faces, filename)
        


    def get_boundary(self):
        x_min = sys.maxsize
        x_max = -sys.maxsize
        y_min = x_min
        z_min = x_min
        y_max = x_max
        z_max = x_max

        lin_indx = 0
        for k in range(self.grid_dim[2]):
            for j in range(self.grid_dim[1]):
                for i in range(self.grid_dim[0]):
                    if torch.abs(self.dist[lin_indx]) > torch.sqrt(torch.Tensor([3.])).to(self.device)*self.voxel_size:
                        lin_indx += 1
                        continue
                    if (self.w[lin_indx] < 0.6):
                        lin_indx += 1
                        continue
                    if (i<x_min): 
                        x_min = i    
                    if (i > x_max):
                        x_max = i
                    if (j < y_min):
                        y_min = j
                    if (j > y_max):
                        y_max = j
                    if (k < z_min):
                        z_min = k
                    if (k > z_max):
                        z_max = k
                    lin_indx += 1

        print("valid dim: {0} {1} {2}.\n".format(x_max-x_min, y_max-y_min, z_max-z_min))

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    # need to improve this 
    def export_gradient_sdf(self, filename):
        # x_min, x_max, y_min, y_max, z_min, z_max = boundary

        with open(filename, 'w') as f:
            f.write("# grid dim {0}\n".format(self.grid_dim))
            f.write("# voxel size {0} \n".format(self.voxel_size))
            f.write("# truncate {0} \n".format(self.T))
            f.write("\n".join([" ".join(["%f %f %f %f %f %f %f %f" % (self.dist[i], self.w[i], self.grad[i,0], self.grad[i,1], self.grad[i,2], self.color[i,0], self.color[i,1], self.color[i,2]) for i in range(self.num_voxels)])]))
        f.close()