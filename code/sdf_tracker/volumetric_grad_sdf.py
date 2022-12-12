import torch
import torch.nn.functional as F
import cv2 as cv
import sys

from sdf_tracker.voxel_grid import *
from sdf_tracker.sdf import *
from third.normal_estimator import *
import third.general as utils

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
        grid_dim,
        z_min=0.5,
        z_max=3.5,
        voxel_size=0.02,
        shift=torch.Tensor([[0.0, 0.0, 0.0]]),
        counter=0
    ):
        VoxelGrid.__init__(self, grid_dim, voxel_size, shift)

        Sdf.__init__(self, T=T*voxel_size, z_min=z_min, z_max=z_max, counter=counter)

        self.Nest = normal_estimator
        # self.tsdf_ = []
        self.vis = torch.ones(self.num_voxels, 1).type(torch.int32)

        self.dist = -torch.ones(self.num_voxels) * self.T
        self.grad = torch.zeros_like(self.grid_points)
        self.color = torch.zeros_like(self.grid_points)
        self.w = torch.zeros(self.num_voxels)
    
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

        tsdf = torch.where(valid, tsdf, torch.Tensor([self.T]))

        return tsdf, normal, valid, index

    # compute the new weight
    def weights(self, points):
        index = self.nearest_index(points)
        valid = (index > 0)
        w = self.w[index]
        
        w = torch.where(valid, w, torch.Tensor([0.0]))

        return w
    
    def getVoxel(self, index):
        return self.tsdf_[self.idx2line(index)]


    def compute_centroid(self, K, depth):
        h, w = depth.shape
        u, v = torch.meshgrid(torch.linspace(0,w-1, w), torch.linspace(0, h-1, h))
        u = u.T.flatten()
        v = v.T.flatten()
        x0 = (u - K[0,2]) / K[0,0]
        y0 = (v - K[1,2]) / K[1,1]

        z = torch.from_numpy(depth.flatten())

        valid = (z > self.z_min) & (z< self.z_max)
        z = torch.where(valid, z, torch.Tensor([0.0]))
        points = torch.vstack([x0*z, y0*z, z]).T

        counter = torch.sum(valid)

        centroid = torch.sum(points, dim=0) / counter

        self.shift = centroid # 0.0248 0.0205 3.69
        self.origin = self.shift - 0.5 * self.voxel_size * torch.Tensor(self.grid_dim).type(torch.float32)
        print("...computed the centroid of first frame: {0} ".format(centroid))

        return centroid


    def update(self, rgb, depth, K, pose):

        med_depth = cv.medianBlur(depth, 5)

        Nest = self.Nest.compute(depth=med_depth)

        grad = Nest['normal']
        n_sq_inv = Nest['n_sq_inv']

        R = pose[:3,:3]
        t = pose[:3,3]

        points = self.voxel2world(self.grid_points)
        points = (points - t.unsqueeze(-2)) @ R

        project_points = points @ K.T
        n = torch.round(project_points[:,0]/project_points[:,2]).type(torch.int64)
        m = torch.round(project_points[:,1]/project_points[:,2]).type(torch.int64)

        valid = (n>=0) & (m>=0) & (n<depth.shape[1]) & (m<depth.shape[0])

        nn = torch.where(valid, n, -1)
        mm = torch.where(valid, m, -1)

        z = med_depth[mm,nn]       
        color = torch.from_numpy(rgb[mm,nn,:])

        valid_z = (z>self.z_min) & (z < self.z_max)

        # Debug: 
        sdf = torch.from_numpy(z) - points[:,2]
        w = self.weight(sdf)

        # trucate sdf
        sdf = self.truncate(sdf)

        valid_weight = (w>0)

        normal = torch.from_numpy(grad[:,mm,nn]).type(torch.float32)
        n_sq_inv_points = torch.from_numpy(n_sq_inv[mm,nn])

        # normal norm smaller than 0.1 is invalide
        valid_n = torch.norm(normal, p=2, dim=0)>=0.1
        xy_hom =  points * (1/points[:,2]).view(points.shape[0], 1)

        # normal angle is less then 75 degree
        valid_angle = torch.sum(normal.T*xy_hom,dim=1)*torch.sum(normal.T*xy_hom,dim=1)*n_sq_inv_points > (0.15*0.15)

        valid = (valid  & valid_weight & valid_z & valid_n) # & valid_angle)

        # Debug: 
        print("-----DEBUG: valid grid number: ", torch.sum(valid))

        # prepare for update
        w = torch.where(valid, w, torch.Tensor([0.]))
        
        # update voxel properties
        self.w += w
        self.dist = torch.where(valid, self.dist + (sdf - self.dist) * w / self.w, self.dist)
        self.grad = torch.where(valid.unsqueeze(-1), self.grad- (R @ normal).T * w.unsqueeze(-1), self.grad)
        self.color = torch.where(valid.unsqueeze(-1), self.color + (color - self.color) * w.unsqueeze(-1) / self.w.unsqueeze(-1), self.color)
        
        # #debug
        # if check_nan(self.grad):
        #     print("there is nan in self grad.")
        
        if self.counter:
            vis = torch.ones(self.num_voxels, 1).type(torch.int32)
            vis[:,0] = torch.where(valid, vis[:,0], torch.Tensor([0]).type(torch.int32))
            self.vis = torch.hstack([self.vis, vis])
        else:
            self.vis[:,0]= torch.where(valid, self.vis[:,0], torch.Tensor([0]).type(torch.int32))


        print('-----counter: ', self.counter)
        print('-----DEBUG: visbility length: ', self.vis.shape[1])

        
    def setup(self, rgb, depth, K):
        pose = torch.eye(4)
        self.update(rgb, depth, K, pose)


    def export_pc(self, filename):
        normal = F.normalize(self.grad, p=2, dim=1)
        points = self.voxel2world(self.grid_points) - self.dist.view(self.num_voxels, 1) * normal
        colors = (self.color * 255).type(torch.int32)
        valid = (self.w > 0.0) & (torch.abs(self.dist) < torch.sqrt(torch.Tensor([3.0]))*self.voxel_size)
        count = torch.sum(valid)
        
        try: 
            with open(filename, 'w') as f:
                f.write( "ply \n")
                f.write( "format ascii 1.0 \n")
                f.write( "element vertex %d \n" % count)
                f.write( "property float x \n")
                f.write( "property float y \n")
                f.write( "property float z \n")
                f.write( "property float nx \n")
                f.write( "property float ny \n")
                f.write( "property float nz \n")
                f.write( "property uchar red \n")
                f.write( "property uchar green \n")
                f.write( "property uchar blue \n")
                f.write( "end_header \n")
            
                for i in range(valid.shape[0]):
                    if valid[i]:
                        f.write(" %f %f %f %f %f %f %d %d %d \n" % (points[i,0], points[i,1], points[i,2], normal[i,0], normal[i,1], normal[i,2], colors[i,0], colors[i,1], colors[i,2]))
        
            f.close()
            return True
        except:
            print("can't save point cloud.")
            return False

    
    def export_mesh(self, boundary, filename):
        x_min, x_max, y_min, y_max, z_min, z_max = boundary
        volumn_ = self.dist.reshape((self.grid_dim[0], self.grid_dim[1], self.grid_dim[2])).permute(2,1,0)
        volumn = volumn_[x_min:x_max, y_min:y_max, z_min:z_max]
    
        utils.export_mesh(volumn, filename)


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
                    if torch.abs(self.dist[lin_indx]) > torch.sqrt(torch.Tensor([3.]))*self.voxel_size:
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