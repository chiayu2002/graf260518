import numpy as np
import torch
from .utils import sample_on_sphere, look_at, to_sphere


def sample_select_pose(u, v):   #計算旋轉矩陣(相機姿勢)
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        loc = to_sphere(u, v)
        
        # sample radius if necessary
        radius = 3.0

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))

        return RT

