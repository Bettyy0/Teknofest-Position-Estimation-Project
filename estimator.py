import numpy as np
from collections import deque
from config import EstimatorConfig, VOConfig
from tracker import KeyframeTracker, Motion, _rot

class PositionEstimator:
    def __init__(self, est_cfg: EstimatorConfig, vo_cfg: VOConfig):
        self.cfg = est_cfg
        self.vo_cfg = vo_cfg
        self.tracker = KeyframeTracker(vo_cfg)
        self.P = np.zeros(3)          
        self.P0 = np.zeros(3)         
        self.psi = 0.0                
        self.h = 1.0                  
        self.vel = np.zeros(3)        
        self.prev_S = (1.0, 0.0, 0.0, 0.0)
        self.have_prev = False
        self.Fp_cum = np.zeros(3)     
        self.Fm_cum = np.zeros(3)     
        self.hist: deque = deque(maxlen=est_cfg.buffer_size)   
        self.samples: deque = deque(maxlen=est_cfg.buffer_size)
        self.A: np.ndarray | None = None
        self.sign = +1
        self.sign_locked = False
        self.H0: float | None = None
        self.alt_axis: np.ndarray | None = None
        self.last_residual = float("nan")
        self.prev_gray: np.ndarray | None = None
        self.frames_seen = 0
        self.n_dr = 0
        self.n_lost = 0
        self.last_status = "init"
        self.locked_z = None  
        self.last_healthy_snap = None  

    def process(self, image: np.ndarray, gt, healthy: bool) -> np.ndarray:
        self.frames_seen += 1
        gt = None if gt is None else np.asarray(gt, dtype=np.float64)
        gray = self.tracker.preprocess(image)
        blank = self.tracker.is_blank(gray, self.vo_cfg.blank_std_thresh)
        frozen = self.tracker.is_frozen(self.prev_gray, gray, self.vo_cfg.frozen_diff_thresh)

        if not self.tracker.has_keyframe:
            if not blank:
                self._new_keyframe(gray)
            self.prev_gray = gray
            self._push_hist(gt, healthy)
            return self._output(gt, healthy)

        fp = fm = None
        m = Motion()

        if blank:
            self.last_status = "blank"                 
        elif frozen:
            self.last_status = "frozen"                
            fp = fm = np.zeros(3)
            self._accumulate(fp, fm)
        else:
            m = self.tracker.track(gray)
            if m.ok:
                self.last_status = "vo"
                out = self._increment_feature(m)
                if out is not None:
                    fp, fm = out
            else:
                self.last_status = "lost"

        if fp is not None and self.A is not None:
            f = fp if self.sign > 0 else fm
            P_new = self.P + self._clip(self.A @ f)
        elif fp is not None:
            P_new = self.P.copy()                      
        else:
            if self.last_status in ("blank", "lost"):
                self.n_lost += 1
                self.tracker.kf_pts = None             
                self.have_prev = False
            P_new = self.P + self.vel                  

        if healthy and gt is not None:
            self.vel = 0.5 * self.vel + 0.5 * (gt - self.P)
            self.P = gt.copy()
            self._reanchor_altitude(gt)
            self.locked_z = float(gt[2]) 
        else:
            self.n_dr += 1
            self.vel = ((1 - self.cfg.vel_smooth) * self.vel
                        + self.cfg.vel_smooth * (P_new - self.P))
            if self.cfg.lock_altitude_in_blind and self.locked_z is not None:
                P_new[2] = self.locked_z  
                self.vel[2] = 0.0         
            self.P = P_new

        self._push_hist(gt, healthy)
        self._collect_samples(healthy)
        if (self.frames_seen % self.cfg.refit_every) == 0:
            self._fit()
        if not blank and (not self.tracker.has_keyframe or self.tracker.needs_new_keyframe(m)):
            self._new_keyframe(gray)
        self.prev_gray = gray
        return self._output(gt, healthy)

    def _increment_feature(self, m: Motion):
        s_k, th_k, tx_k, ty_k = m.scale, m.dpsi, m.tx, m.ty
        if not self.have_prev:
            self.prev_S = (s_k, th_k, tx_k, ty_k)
            self.have_prev = True
            return None
        s_p, th_p, tx_p, ty_p = self.prev_S
        ds = s_k / max(s_p, 1e-6)
        dth = th_k - th_p
        dt = np.array([tx_k, ty_k]) - ds * (_rot(dth) @ np.array([tx_p, ty_p]))
        self.prev_S = (s_k, th_k, tx_k, ty_k)
        psi_mid = self.psi + 0.5 * dth
        h_prev = self.h
        h_raw = h_prev / max(ds, 1e-6)
        
        if self.H0 is not None and self.H0 > 1e-6:
            dh_max = self.cfg.max_speed_v * self.cfg.dt / self.H0
        else:
            dh_max = 0.02
        h_raw = float(np.clip(h_raw, h_prev - dh_max, h_prev + dh_max))
        
        if self.cfg.freeze_h_in_blind and not self.prev_ref_healthy:
            h_new = h_prev
        else:
            h_new = (1 - self.cfg.h_lp) * h_raw + self.cfg.h_lp * h_prev
            
        self.psi += dth
        self.h = h_new
        dz = h_new - h_prev
        up = _rot(+psi_mid) @ dt * h_prev
        um = _rot(-psi_mid) @ dt * h_prev
        fp = np.array([up[0], up[1], dz])
        fm = np.array([um[0], um[1], dz])
        self._accumulate(fp, fm)
        return fp, fm

    def _accumulate(self, fp, fm):
        self.Fp_cum = self.Fp_cum + fp
        self.Fm_cum = self.Fm_cum + fm

    def _push_hist(self, gt, healthy: bool):
        self.hist.append((
            self.frames_seen,
            self.Fp_cum.copy(), 
            self.Fm_cum.copy(),
            gt.copy() if (healthy and gt is not None) else None
        ))

    def _collect_samples(self, healthy: bool):
        if not healthy or self.hist[-1][3] is None:
            return
        idx_k, Fp_k, Fm_k, gt_k = self.hist[-1]
        for W in self.cfg.windows:
            if len(self.hist) <= W: continue
            idx_0, Fp_0, Fm_0, gt_0 = self.hist[-1 - W]
            if gt_0 is None or (idx_k - idx_0) != W: continue
            self.samples.append((Fp_k - Fp_0, Fm_k - Fm_0, gt_k - gt_0))

        if self.last_healthy_snap is not None:
            i0, Fp0, Fm0, g0 = self.last_healthy_snap
            gap = self.frames_seen - i0
            if gap >= 30: 
                self.samples.append((self.Fp_cum - Fp0, self.Fm_cum - Fm0, gt_k - g0))
                
        self.last_healthy_snap = (self.frames_seen, self.Fp_cum.copy(), self.Fm_cum.copy(), gt_k.copy())

    @staticmethod
    def _ridge(F, D, w, lam_rel):
        sw = np.sqrt(w)[:, None]
        Fw, Dw = F * sw, D * sw
        G = Fw.T @ Fw
        lam = lam_rel * (np.trace(G) / 3.0 + 1e-12)
        return np.linalg.solve(G + lam * np.eye(3), Fw.T @ Dw).T   

    def _fit_constrained(self, F, D, w, iters=6):
        A0 = self._ridge(F, D, w, self.cfg.ridge_rel)
        U, S, Vt = np.linalg.svd(A0[:, :2], full_matrices=False)
        Uc = U @ Vt                                  
        n = np.cross(Uc[:, 0], Uc[:, 1])
        nn = np.linalg.norm(n)
        if nn < 1e-9: return A0, float("inf")
        
        Q = np.column_stack([Uc[:, 0], Uc[:, 1], n / nn])
        k = float(S.mean())
        H0 = float(np.linalg.norm(A0[:, 2])) or 1.0

        for _ in range(iters):
            u1, u2, u3 = Q[:, 0], Q[:, 1], Q[:, 2]
            Ak = F[:, 0:1] * u1 + F[:, 1:2] * u2      
            An = F[:, 2:3] * u3                       
            dk = float(np.sum(w * (Ak ** 2).sum(1)))
            dn = float(np.sum(w * (An ** 2).sum(1)))
            if dk > 1e-12: k = float(np.sum(w * (D * Ak).sum(1)) / dk)
            if dn > 1e-12: H0 = float(np.sum(w * (D * An).sum(1)) / dn)

            G = F * np.array([k, k, H0])
            M = (D * w[:, None]).T @ G
            Uu, _, Vv = np.linalg.svd(M)
            Q = Uu @ Vv                               

        if H0 < 0:                                    
            H0 = -H0
            Q[:, 2] = -Q[:, 2]
        A = Q @ np.diag([k, k, H0])
        res = float(np.sum(((F @ A.T - D) ** 2).sum(1) * w) / max(w.sum(), 1e-9))
        return A, res

    def _fit(self):
        n = len(self.samples)
        if n < self.cfg.min_samples: return
        Fp = np.stack([s[0] for s in self.samples])
        Fm = np.stack([s[1] for s in self.samples])
        D = np.stack([s[2] for s in self.samples])
        w = self.cfg.forget ** np.arange(n)[::-1]
        cands = ([(self.sign, Fp if self.sign > 0 else Fm)] if self.sign_locked
                 else [(+1, Fp), (-1, Fm)])
        best = None
        for sg, F in cands:
            A, res = self._fit_constrained(F, D, w)
            if best is None or res < best[1]:
                best = (A, res, sg)
        if best is None: return
        A, self.last_residual, sg = best
        self.A = A
        if not self.sign_locked:
            self.sign = sg
            if n >= self.cfg.sign_lock_samples:
                self.sign_locked = True
        col = A[:, 2]
        nrm = float(np.linalg.norm(col))
        if nrm > 1e-9:
            self.H0 = nrm
            self.alt_axis = col / nrm

    def _reanchor_altitude(self, gt: np.ndarray):
        if not self.cfg.reanchor_altitude or self.H0 is None or self.alt_axis is None: return
        if not (self.cfg.h0_min < self.H0 < self.cfg.h0_max): return
        alt = float(self.alt_axis @ (gt - self.P0))
        h_gt = 1.0 + alt / self.H0
        if 0.2 < h_gt < 5.0:
            self.h = 0.5 * self.h + 0.5 * h_gt

    def _new_keyframe(self, gray: np.ndarray):
        self.tracker.reset(gray)
        self.prev_S = (1.0, 0.0, 0.0, 0.0)   
        self.have_prev = True
        
    @property
    def prev_ref_healthy(self):
        return self.last_status != "init" and getattr(self, "_prev_ref_healthy", False)
        
    @prev_ref_healthy.setter
    def prev_ref_healthy(self, value):
        self._prev_ref_healthy = value

    def _clip(self, dP: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt
        hmax = self.cfg.max_speed_h * dt
        vmax = self.cfg.max_speed_v * dt
        if self.alt_axis is not None:
            w = self.alt_axis
            dv = float(w @ dP)
            dh = dP - dv * w
            nh = float(np.linalg.norm(dh))
            if nh > hmax: dh = dh * (hmax / nh)
            dv = float(np.clip(dv, -vmax, vmax))
            return dh + dv * w
        n = float(np.linalg.norm(dP))
        lim = float(np.hypot(hmax, vmax))
        return dP * (lim / n) if n > lim else dP

    def _output(self, gt, healthy: bool) -> np.ndarray:
        if healthy and gt is not None:
            return gt.astype(np.float64)     
        return self.P.copy()

    @staticmethod
    def to_submission(pos: np.ndarray) -> dict:
        return {
            "translation_x": float(pos[0]),
            "translation_y": float(pos[1]),
            "translation_z": float(pos[2])
        }
