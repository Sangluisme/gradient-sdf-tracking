import os
from datetime import datetime
import time
from pyhocon import ConfigFactory
import argparse
import sys
import torch
import numpy as np

import sdf_tracker.rigid_point_optimizer as RigidOptimizer
import sdf_tracker.volumetric_grad_sdf as VolGradSdf
import third.image_loader as Loader
import third.normal_estimator as NormalEstimator
import third.general as utils
from third.timer import Timer as Timer

_first = 0
_last = sys.maxsize

class Tracker():
    def __init__(self, device, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])

        # create folder to save results
        self.exps_folder_name = kwargs['results_dir']
        self.expname = kwargs['expname']
        self.device = device

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join(self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(os.path.join(self.expdir))

        # parse dataset config
        dataset_conf = self.conf.get_config('dataset')
        if dataset_conf['mode'] == 'tum':
            data_dir = os.path.join(dataset_conf['data_path'], dataset_conf['prefix']+self.expname)
        else:
            data_dir = os.path.join(dataset_conf['data_path'], self.expname)

        self.start = self.conf.get_int('dataset.first', _first)
        self.end = self.conf.get_int('dataset.last', _last)

        intrinsic = self.conf.get_string('dataset.intrinsics', "intrinsics.txt")
        assoc_file = self.conf.get_string('dataset.assoc_file', "associated.txt")
        depth_prefix = self.conf.get_string('dataset.depth_prefix', "depth/")
        rgb_prefix = self.conf.get_string('dataset.rgb_prefix', "rgb/")
        digital = self.conf.get_int('dataset.digital', 3)
        depth_factor = dataset_conf['depth_factor']

        T = Timer()

        T.tic()
        self.loader = Loader.ImageLoader(data_dir=data_dir, 
                                    mode=dataset_conf['mode'], 
                                    intrinsics_file=intrinsic, 
                                    assoc_file=assoc_file,
                                    depth_prefix=depth_prefix,
                                    rgb_prefix=rgb_prefix,
                                    depth_factor=depth_factor,
                                    digital=digital)
        T.toc("initial image loader")
        print("...load dataset intrinsics:\n {0}".format(self.loader.K))
        
        # parse argument to normal estimator
        T.tic()
        self.normal_estimator = NormalEstimator.NormalEstimator(self.loader.K, **self.conf.get_config('model')['normal_estimator'])
        T.toc("initial normal estimator")
        
        # parse argument for sdf
        truncate = self.conf.get_int('model.grad_sdf.T')
        self.tSDF = VolGradSdf.VolumetricGradSdf(self.normal_estimator, device=self.device,  **self.conf.get_config('model')['grad_sdf'])
        
        print("...initial grid size {0} ".format(self.tSDF.grid_dim))
        print("...initial voxel size is {0} ".format(self.tSDF.voxel_size))

        self.pOpt = RigidOptimizer.RigidPointOptimizer(self.tSDF, torch.eye(4), device=self.device)

        # self.tOpt.check_tsdf()

    def load_image(self, index):
        timestamp, rgb, depth = self.loader.load_img_pair(index)
        return timestamp, rgb, depth



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', type=str, default='../results/', help="result save path.")
    parser.add_argument('--expname', type=str, help='dataset name', required=True)
    parser.add_argument('--conf', type=str, help='config file name.', required=True)
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('running on {0}'.format(device))

    tracker = Tracker(device=device,
                        conf=args.conf,
                        expname=args.expname,
                        results_dir=args.results_dir)
    T = Timer()

    timestamp, rgb, depth = tracker.load_image(tracker.start)
    assert (rgb is not None), "fail to load rgb image."
    assert (depth is not None), "fail to load depth image."
        

    T.tic()
    tracker.tSDF.compute_centroid(tracker.loader.K, depth)
    T.toc("compute shift")

    T.tic()
    tracker.tSDF.setup(rgb, depth, tracker.loader.K)
    T.toc("initial SDF")
    
    pose_f = open(tracker.expdir + "tracking_pose.txt",'w')
    pose_f.write(("%s" % timestamp) + " ".join(["%f"%v  for v in tracker.pOpt.pose.Quaternion().tolist()]) + "\n")
    tracking_pose = []
   
    for i in range(tracker.end-2):
        timestamp, rgb, depth = tracker.load_image(i+2)
        if ((rgb is None) or (depth is None)):
            print("couldn't load depth or rgb images.")
            break

        tracker.tSDF.update_counter()

        T.tic()
        conv = tracker.pOpt.optimize(depth, tracker.loader.K)
        T.toc("point optimization")

        if conv:
            T.tic()
            tracker.tSDF.update(rgb, depth, tracker.loader.K, tracker.pOpt.pose.mat)
            T.toc('integrate depth data to sdf')
        
        print("current pose:\n {0}".format(tracker.pOpt.pose.mat))
        pose_f.write(("%s " % timestamp) + " ".join(["%f "%v  for v in tracker.pOpt.pose.Quaternion().tolist()]) + "\n")
        tracking_pose.append(tracker.pOpt.pose)

        #save middle pc/mesh
        if (i+2) % 50 == 0:
            tracker.tSDF.export_pc(tracker.expdir + "checkpoint_" + str(i+2) + ".ply")

    pose_f.close()

    # save pointcloud
    T.tic()
    tracker.tSDF.export_pc(tracker.expdir + "init_pc.ply")
    T.toc("save pointcloud")

    # save mesh
    # Note that the marching cube function works not as good as c++ one
    T.tic()
    # boundary = tracker.tSDF.get_boundary()
    tracker.tSDF.export_mesh(tracker.expdir + "init_mesh.ply")
    T.toc("save mesh")



