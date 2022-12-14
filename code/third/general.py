import os
from glob import glob
import torch
import numpy as np
from skimage import measure
# import trimesh

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def load_timestamp_file(filename):
    """
    specially for format like tum rgbd data set

    Output
    list -- has the format [timestamp_rgb rgb_filename timestamp_depth depth_filename]
    """

    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[str(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]

    return list



def load_intrinsics(filename):
    file = open(filename)
    data = file.read()
    lines = [ [float(v.strip()) for v in item.split()] for item in data.split('\n')[:-1]]
    return np.asarray(lines)



def initial_grid(grid_dim):      
    xx, yy, zz = np.mgrid[0:grid_dim[0], 0:grid_dim[1], 0:grid_dim[2]]
    grid_points = torch.tensor(np.vstack([xx.flatten('F'), yy.flatten('F'), zz.flatten('F')]).T, dtype=torch.float)
    return xx, yy, zz, grid_points




def export_mesh(sdf, filename):

    verts, faces, normals, values = measure.marching_cubes(sdf.numpy(), level=0) #, method='lewiner', gradient_direction='ascent')



    if verts.shape[0] == 0:
        return False
    
    with open(filename, 'w') as f:

        f.write( "ply \n")
        f.write( "format ascii 1.0 \n")
        f.write( "element vertex %d \n" % verts.shape[0])
        f.write( "property float x \n")
        f.write( "property float y \n")
        f.write( "property float z \n")
        # f.write( "property uchar red")
        # f.write( "property uchar green")
        # f.write( "property uchar blue")
        f.write( "element face %d \n" % faces.shape[0])
        f.write( "property list uchar int vertex_indices \n")
        f.write( "end_header \n")

        # write vertices
        for i in range(verts.shape[0]):
            f.write( "%f %f %f \n" % (verts[i][0], verts[i][1], verts[i][2]))
        
        # for i in range(color.shape[0]):
        #     f.write( "%f %f %f \n" % (color[i][0], color[i][1], color[i][2]))

        # write faces
        for i in range(faces.shape[0]):
            f.write( "3 %d %d %d \n" % (faces[i][0], faces[i][1], faces[i][2]))
        

    f.close()
    return True




