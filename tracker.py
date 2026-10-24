import math
import cv2
import numpy as np
from config import VOConfig

class Motion:
    def __init__(self):
        self.ok = False
        self.scale = 1.0
        self.dpsi = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.tracked_pts = 0             

class KeyframeTracker:
    def __init__(self, cfg: VOConfig):
        self.cfg = cfg
        self.has_keyframe = False
        self.kf_gray = None
        self.kf_pts = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.mask = None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        if self.mask is None or self.mask.shape != img.shape:
            self.mask = np.ones_like(img, dtype=np.uint8) * 255
            if self.cfg.mask_top_ratio > 0.0:
                h = img.shape[0]
                self.mask[:int(h * self.cfg.mask_top_ratio), :] = 0
        img_clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip, tileGridSize=(8, 8)).apply(img)
        return cv2.bitwise_and(img_clahe, img_clahe, mask=self.mask)

    def is_blank(self, gray: np.ndarray, thresh: float) -> bool:
        return gray.std() < thresh

    def is_frozen(self, prev: np.ndarray, curr: np.ndarray, thresh: float) -> bool:
        if prev is None: return False
        return cv2.absdiff(prev, curr).mean() < thresh

    def _detect_gftt_grid(self, gray):
        pts = []
        gh, gw = 4, 6
        h, w = gray.shape
        per = self.cfg.gftt_max_feat // (gh * gw)
        for i in range(gh):
            for j in range(gw):
                cell = gray[i*h//gh:(i+1)*h//gh, j*w//gw:(j+1)*w//gw]
                kps = cv2.goodFeaturesToTrack(cell, maxCorners=per, 
                                              qualityLevel=self.cfg.gftt_quality, 
                                              minDistance=self.cfg.gftt_min_dist, blockSize=3)
                if kps is not None:
                    for k in kps:
                        pts.append([k[0][0] + j*w//gw, k[0][1] + i*h//gh])
        if not pts: return None
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def reset(self, gray: np.ndarray):
        self.kf_gray = gray
        self.kf_pts = self._detect_gftt_grid(gray)
        self.has_keyframe = self.kf_pts is not None and len(self.kf_pts) > 50

    def track(self, gray: np.ndarray) -> Motion:
        m = Motion()
        if not self.has_keyframe or self.kf_pts is None: return m
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.kf_gray, gray, self.kf_pts, None, **self.lk_params)
        if p1 is None: return m
        p0r, st2, err2 = cv2.calcOpticalFlowPyrLK(gray, self.kf_gray, p1, None, **self.lk_params)
        if p0r is None: return m
        fb_dist = np.linalg.norm(self.kf_pts.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_dist < self.cfg.fb_thresh)
        q0 = self.kf_pts[good]
        q1 = p1[good]
        m.tracked_pts = len(q0)
        if len(q0) < 20: return m

        q0_norm = cv2.undistortPoints(q0, self.cfg.K, self.cfg.D).reshape(-1, 2)
        q1_norm = cv2.undistortPoints(q1, self.cfg.K, self.cfg.D).reshape(-1, 2)
        ransac_norm_thresh = self.cfg.ransac_thresh / self.cfg.fx
        H, inl = cv2.findHomography(q0_norm, q1_norm, cv2.RANSAC, ransac_norm_thresh, maxIters=1000, confidence=0.995)
        
        if H is not None and inl is not None and inl.sum() > 0.5 * len(q0):
            h_img, w_img = gray.shape
            x = np.array([0.0, 0.0, 1.0])
            w_h = H @ x
            p = w_h[:2] / w_h[2]
            J = (H[:2, :2] * w_h[2] - np.outer(w_h[:2], H[2, :2])) / (w_h[2] ** 2)
            U, S, Vt = np.linalg.svd(J)
            R = U @ Vt
            s = float(np.sqrt(max(np.linalg.det(J), 1e-9)))
            m.ok = True
            m.scale = s
            m.dpsi = math.atan2(R[1, 0], R[0, 0])
            m.tx = float(p[0])
            m.ty = float(p[1])
        else:
            M, inliers = cv2.estimateAffinePartial2D(q0, q1, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_thresh)
            if M is not None:
                m.ok = True
                m.tx = M[0, 2]
                m.ty = M[1, 2]
                a = M[0, 0]
                b = M[1, 0]
                m.scale = math.hypot(a, b)
                m.dpsi = math.atan2(b, a)
        return m

    def needs_new_keyframe(self, m: Motion) -> bool:
        if not m.ok: return True
        if m.tracked_pts < self.cfg.min_tracked_pts: return True
        if m.scale < 0.60 or m.scale > 1.65: return True  
        if abs(m.dpsi) > 0.60: return True               
        if math.hypot(m.tx, m.ty) > 0.2: return True    
        return False

def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)
