import os
import numpy as np
import argparse

from third.camera_utils import *
import third.general as utils

class ImageLoader:
    """
    load RGB-D data sequence

    """

    def __init__(
        self, 
        data_dir,
        mode,
        intrinsics_file,
        assoc_file=None,
        depth_prefix=None,
        rgb_prefix=None,
        depth_factor=1000,
        digital=3,
        pose_file=None
        ):
        
        self.data_dir_ = data_dir
        self.mode_ = mode
        self.assoc_ = assoc_file
        self.digital_ = digital
        self.rgb_prefix_ = rgb_prefix
        self.depth_prefix_ = depth_prefix
        self.depth_factor_ = depth_factor

        self.K = utils.load_intrinsics(os.path.join(data_dir, intrinsics_file))

        self.name_list_ = []

        if self.mode_ == 'tum':
            assert len(self.assoc_)>0, "associate file name is missing"
            self.depth_factor_ = 5000
            self.assoc_file_ = os.path.join(data_dir, self.assoc_)
            self.name_list_ = utils.load_timestamp_file(self.assoc_file_)

        else:
            assert depth_prefix is not None, "depth_format should be given"
            assert rgb_prefix is not None, "rgb_format should be given"
        
            
    def img_path(self, indx):
        if self.mode_ == 'tum':
            rgb_path = os.path.join(self.data_dir_, self.name_list_[indx][1])
            depth_path = os.path.join(self.data_dir_, self.name_list_[indx][3])
            rgb_timestamp = self.name_list_[indx][0]
            depth_timestamp = self.name_list_[indx][2]
        
        else:
            number = str(indx).zfill(self.digital_) + '.png'
            rgb_path = os.path.join(self.data_dir_, self.rgb_prefix_, number)
            depth_path = os.path.join(self.data_dir_, self.depth_prefix_, number)
            rgb_timestamp = str(indx).zfill(self.digital_)
            depth_timestamp = rgb_timestamp

        return rgb_timestamp, depth_timestamp, rgb_path, depth_path

    
    def load_img_pair(self, indx):
        _, timestamp, rgb_file, depth_file = self.img_path(indx)
        print('rgb image file name:', rgb_file)
        print('depth image file name:', depth_file)
        depth = load_depth(depth_file, self.depth_factor_)
        rgb = load_rgb(rgb_file)

        return timestamp, rgb, depth


# debug
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'DataLoader',
                    description = 'load data from different dataset with different format',
                    epilog = 'Text at the bottom of help')

    parser.add_argument('--data_dir', type=str, help='data folder path.', required=True)
    parser.add_argument('--mode', type=str, help='tum or others', required=True)
    parser.add_argument('--intrinsics', type=str, default='intrinsics.txt', help='intrinsics file name')
    parser.add_argument('--assoc', type=str, default="associated.txt", help='when mode is tum, associates file is needed.')
    parser.add_argument('--depth_prefix', type=str, default='depth/', help='depth prefix when mode is others')
    parser.add_argument('--rgb_prefix', type=str, default='rgb/', help='rgb prefix when mode is others.')
    parser.add_argument('--digital', type=int, help='how many digitals in the name of images, when mode is others')
    parser.add_argument('--index', type=int, default=0, help='which image to load.')


    args = parser.parse_args()

    img_loader = ImageLoader(data_dir=args.data_dir, 
                            mode=args.mode,
                            intrinsics_file=args.intrinsics,
                            assoc_file=args.assoc,
                            depth_prefix=args.depth_prefix,
                            rgb_prefix=args.rgb_prefix,
                            digital=args.digital)

    rgb, depth = img_loader.load_img_pair(args.index)

    print(rgb.shape, depth.shape)

