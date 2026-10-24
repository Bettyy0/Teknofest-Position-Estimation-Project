import cv2
import numpy as np
from config import EstimatorConfig, VOConfig
from estimator import PositionEstimator
from api import TeknofestAPIWrapper

def main():
    print("TEKNOFEST 2026 Havacılıkta Yapay Zeka - Vektör Takımı")
    print("Pozisyon Kestirim Modülü Başlatılıyor...")
    
    est_cfg = EstimatorConfig()
    vo_cfg = VOConfig()
    estimator = PositionEstimator(est_cfg, vo_cfg)
    api = TeknofestAPIWrapper(estimator)
    
    # Örnek dummy frame
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    json_sonuc = api.predict(
        frame=dummy_frame,
        is_healthy=True,
        frame_url="http://example.com/frame.jpg",
        submission_id=1,
        user_url="http://localhost/users/1/",
        gt_x=10.0, gt_y=20.0, gt_z=30.0
    )
    
    print("Çıktı Başarıyla Üretildi:\n", json_sonuc)

if __name__ == "__main__":
    main()
