import cv2 as cv
import numpy as np
# import general as utils
import torch

def check_nan(array):
    if(np.isnan(array).any()):
        print("nan found here.")

class NormalEstimator:
    def __init__(
        self,
        K,
        img_res,
        window_size=5
        ):

        self.img_res_ = img_res
        self.window_size_ = window_size * 2 + 1
        self.K = K
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.fx = K[0,0]
        self.fy = K[1,1]

        self.cache()

    def pixelgrid(self):
        u, v = np.mgrid[0:self.img_res_[1], 0:self.img_res_[0]].astype(np.float32)
        u = u.T - self.cx
        v = v.T - self.cy

        return u, v

    def cache(self):
        fx_inv = 1./self.fx
        fy_inv = 1./self.fy
        
        x0, y0 = self.pixelgrid()

        x0 = fx_inv * x0
        x0_sq = x0 * x0
        y0 = fy_inv * y0
        y0_sq = y0 * y0
        x0_y0 = x0 * y0

        n_sq = 1.0 + x0_sq + y0_sq
        n_sq_inv = 1 / n_sq
        x0_n_sq_inv = x0 * n_sq_inv
        y0_n_sq_inv = y0 * n_sq_inv

        
        M11 = cv.boxFilter(x0_sq*n_sq_inv, -1, ksize=(self.window_size_, self.window_size_),anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        M12 = cv.boxFilter(x0_y0*n_sq_inv, -1,  ksize=(self.window_size_, self.window_size_),anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        M13 = cv.boxFilter(x0_n_sq_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        M22 = cv.boxFilter(y0_sq*n_sq_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        M23 = cv.boxFilter(y0_n_sq_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        M33 = cv.boxFilter(n_sq_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)

        det = M11 * M22 * M33 + 2 * M12 * M23 * M13 - (M13 * M13 * M22) - (M12 * M12 * M33) - (M23 * M23 * M11)

        det_inv = 1.0 / det
        det_inv = np.where(det==0, 0.0, det_inv)

        Q11 = det_inv * (M22 * M33 - M23 * M23)
        Q12 = det_inv * (M13 * M23 - M12 * M33)
        Q13 = det_inv * (M12 * M23 - M13 * M22)
        Q22 = det_inv * (M11 * M33 - M13 * M13)
        Q23 = det_inv * (M12 * M13 - M11 * M23)
        Q33 = det_inv * (M11 * M22 - M12 * M12)

        self.x0 = x0
        self.y0 = y0
        self.x0_n_sq_inv = x0_n_sq_inv
        self.y0_n_sq_inv = y0_n_sq_inv
        self.n_sq_inv = n_sq_inv
        self.Q11 = Q11
        self.Q12 = Q12
        self.Q13 = Q13
        self.Q22 = Q22
        self.Q23 = Q23
        self.Q33 = Q33
       
        # check_nan(Q11)
        # check_nan(Q12)

    def compute(self, depth):
    
        tmp = 1 / depth
        z_inv = np.where(depth==0, 0, tmp)

        b1 = cv.boxFilter(self.x0_n_sq_inv*z_inv, -1,  ksize=(self.window_size_, self.window_size_),anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)
        b2 = cv.boxFilter(self.y0_n_sq_inv*z_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1),normalize=False, borderType=cv.BORDER_DEFAULT)
        b3 = cv.boxFilter(self.n_sq_inv*z_inv, -1,  ksize=(self.window_size_, self.window_size_), anchor=(-1,-1), normalize=False, borderType=cv.BORDER_DEFAULT)

        
        nx = b1 * self.Q11 + b2 * self.Q12 + b3 * self.Q13
        ny = b1 * self.Q12 + b2 * self.Q22 + b3 * self.Q23
        nz = b1 * self.Q13 + b2 * self.Q23 + b3 * self.Q33

        norm_n = np.sqrt(nx * nx + ny * ny + nz * nz)
        
        norm_n_inv = 1 / norm_n
        norm_n_inv = np.where(norm_n==0, 0, norm_n_inv)

        nx = nx * norm_n_inv
        ny = ny * norm_n_inv
        nz = nz * norm_n_inv


        normal = np.stack((nx,ny,nz))

        # for j in range(nx.shape[1]):
        #     for i in range(nx.shape[0]):
        #         if norm_n[i,j]>0:
        #             print(nx[i,j], ny[i,j], nz[i,j])

        Nest = {
            'normal':normal,
            'n_sq_inv': self.n_sq_inv
        }

        return Nest