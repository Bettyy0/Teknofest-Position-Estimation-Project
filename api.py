import json
import numpy as np
from estimator import PositionEstimator

class TeknofestAPIWrapper:
    def __init__(self, estimator: PositionEstimator):
        self.estimator = estimator

    def predict(self, frame: np.ndarray, is_healthy: bool, frame_url: str = "", submission_id: int = 0, user_url: str = "http://localhost/users/0/", gt_x=None, gt_y=None, gt_z=None) -> str:
        if is_healthy and gt_x is not None and gt_y is not None and gt_z is not None:
            gt_vec = np.array([gt_x, gt_y, gt_z], dtype=np.float64)
        else:
            gt_vec = None
            
        pred_pos = self.estimator.process(frame, gt_vec, is_healthy)
        
        data = {
            "id": submission_id,
            "user": user_url,
            "frame": frame_url,
            "detected_objects": [], 
            "detected_translations": [
                {
                    "translation_x": float(pred_pos[0]),
                    "translation_y": float(pred_pos[1]),
                    "translation_z": float(pred_pos[2])
                }
            ],
            "detected_undefined_objects": [] 
        }
        
        return json.dumps(data)
