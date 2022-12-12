import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F
import sys
import os

_EPS = np.finfo(float).eps * 4.0

def load_rgb(path):
    if os.path.isfile(path):
        img = imageio.imread(path)
        img = skimage.img_as_float32(img)
        # load = True
    else:
        img = None
        # load = False

    return img

def load_depth(path, factor):
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = np.array(img).astype(np.float32) / factor
    else:
        img = None
    return img


def load_pose(filename, matrix=True):
    """
    Read a trajectory from a text file. 
    
    Input:
    filename -- file to be read
    matrix -- convert poses to 4x4 matrices
    
    Output:
    dictionary of stamped 3D poses
    """

    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list_ok = []
    for i,l in enumerate(list):
        if l[4:8]==[0,0,0,0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v): 
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n"%(i,filename))
            continue
        list_ok.append(l)
    if matrix :
      traj = dict([(l[0],transform44(l[0:])) for l in list_ok])
    else:
      traj = dict([(l[0],l[1:8]) for l in list_ok])
    return traj


def save_pose(filename, poses, timestampe=[]):
    """
    save poses into quaterion
    Input 
    poses -- n x 4 x 4 torch tensor
    Filename -- saved pose file name

    Output 
    save a txt file
    """
    result = []
    t = poses[:,:3, 3]
    q = rot_to_quat(poses[:,:3,:3])

    if len(timestampe)>0:
        result = dict([(timestampe[l], t[l,:], q[l,:])for l in range(poses.shape[0])])
    else:
        result = dict([(l, t[l,:], q[l,:])for l in range(poses.shape[0])])
    f = open(filename, 'w')
    f.write("\n".join([" ".join(["%f"%v for v in line]) for line in result]))
    f.close()


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)




def rot_to_quat(R):
    # batch_size, _,_ = R.shape
    q = torch.ones((4))

    R00 = R[0, 0]
    R01 = R[0, 1]
    R02 = R[0, 2]
    R10 = R[1, 0]
    R11 = R[1, 1]
    R12 = R[1, 2]
    R20 = R[2, 0]
    R21 = R[2, 1]
    R22 = R[2, 2]

    q[0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[1]=(R21-R12)/(4*q[0])
    q[2] = (R02 - R20) / (4 * q[0])
    q[3] = (R10 - R01) / (4 * q[0])
    return q
