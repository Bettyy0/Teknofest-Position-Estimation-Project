import numpy as np

class EstimatorConfig:
    def __init__(self):
        self.buffer_size = 500           
        self.windows = [15, 30, 45]      
        self.ridge_rel = 1e-4            
        self.vel_smooth = 0.05           
        self.lock_altitude_in_blind = False 
        self.h_lp = 0.15                 
        self.freeze_h_in_blind = True    
        self.min_samples = 20            
        self.forget = 0.99               
        self.sign_lock_samples = 50      
        self.reanchor_altitude = True    
        self.h0_min = 2.0                
        self.h0_max = 500.0              
        self.dt = 1.0 / 7.5              
        self.max_speed_h = 28.0          
        self.max_speed_v = 10.0          
        self.refit_every = 5             

class VOConfig:
    def __init__(self):
        self.blank_std_thresh = 5.0      
        self.frozen_diff_thresh = 0.5    
        self.mask_top_ratio = 0.0        
        self.clahe_clip = 3.0            
        self.gftt_max_feat = 1500        
        self.gftt_quality = 0.02         
        self.gftt_min_dist = 10          
        self.fb_thresh = 1.2             
        self.ransac_thresh = 2.0         
        self.min_tracked_pts = 60        
        self.fast_threshold = 10
        self.K = np.array([[1389.7, 0, 954.007],
                           [0, 1387.1, 558.896],
                           [0, 0, 1]], dtype=np.float64)
        self.D = np.array([0.1378, -0.2564, 0.0, 0.0, 0.0], dtype=np.float64)
        self.fx = self.K[0, 0]
