import os
import numpy as np

class CLSExamplar(object):
    def __init__(self, topo_str, base_dir='cls_matrix'):
        self.A = np.load(os.path.join(os.path.dirname(__file__), base_dir, topo_str + '.npy'))
        