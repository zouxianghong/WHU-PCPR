import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

    
class ScanContext:
    def __init__(self, num_sector=60, num_ring=20, max_length=30.0, lidar_height=2.0):
        self.num_sector = num_sector
        self.num_ring = num_ring
        self.max_length = max_length
        self.lidar_height = lidar_height
        
    def xy2theta(self, x, y):
        if (x >= 0 and y >= 0): 
            theta = 180/np.pi * np.arctan(y/x);
        if (x < 0 and y >= 0): 
            theta = 180 - ((180/np.pi) * np.arctan(y/(-x)));
        if (x < 0 and y < 0): 
            theta = 180 + ((180/np.pi) * np.arctan(y/x));
        if ( x >= 0 and y < 0):
            theta = 360 - ((180/np.pi) * np.arctan((-y)/x));

        return theta
        
    def pt2rs(self, point, gap_ring, gap_sector, num_ring):
        x = point[0]
        y = point[1]
        z = point[2]
        
        if(x == 0.0):
            x = 0.001
        if(y == 0.0):
            y = 0.001
     
        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x*x + y*y)
        
        idx_ring = np.divmod(faraway, gap_ring)[0]
        idx_sector = np.divmod(theta, gap_sector)[0]

        if(idx_ring >= num_ring):
            idx_ring = num_ring-1 # python starts with 0 and ends with N-1
        
        return int(idx_ring), int(idx_sector)
    
    def gen_sc(self, pc_xyz):
        gap_ring = self.max_length/self.num_ring
        gap_sector = 360/self.num_sector
        
        enough_large = 1000
        sc_storage = np.zeros([enough_large, self.num_ring, self.num_sector])
        sc_counter = np.zeros([self.num_ring, self.num_sector])
        
        for pt_idx in range(pc_xyz.shape[0]):
            point = pc_xyz[pt_idx, :]
            point_height = point[2] + self.lidar_height
            
            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, self.num_ring)
            
            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)
        return sc
    
    def gen_sc_parrallel(self, pcs, func, num_processes=None):  # pcs: B x N x 3 or list of N x 3
        if isinstance(pcs, np.ndarray) and pcs.ndim == 3:
            pcs_list = [pcs[i] for i in range(len(pcs))]
        elif isinstance(pcs, list):
            pcs_list = pcs
        else:
            raise ValueError("输入必须是BxNx3的数组或点云列表")
        
        if num_processes is None:
            num_processes = min(cpu_count(), len(pcs_list))

        with Pool(processes=num_processes) as pool:
            scs_list = list(tqdm(
                pool.imap(func, pcs_list),
                total=len(pcs_list),
                desc="Generate ScanContext Descriptors"
            ))
        scs = np.vstack([np.expand_dims(sc, axis=0) for sc in scs_list])
        return scs  # B x num_ring x num_sector

    
if __name__ == "__main__":
    pc_xyz = np.random.random((1024, 3))
    sc = ScanContext()
    bev = sc.gen_sc(pc_xyz)
    print(bev.shape)
