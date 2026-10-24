# Teknofest-Position-Estimation-Project
TEKNOFEST 2026 Artificial Intelligence in Aviation - Position Estimation

This project was developed for the TEKNOFEST Artificial Intelligence in Aviation competition to estimate the position of UAVs/Aircraft during flight. The system utilizes Visual Odometry (VO) and Ridge Regression techniques.

Features

GFTT Grid Detection

Homography-based local similarity calculation

Low-Pass Filters for Altitude (Z)

Drift Protection via Ridge EKF Calibrations

Installation

Clone the repository:

git clone https://github.com/Bettyy0/Teknofest-Position-Estimation-Project


Install the required dependencies:

pip install -r requirements.txt


File Structure

config.py: Contains VO and Estimator configurations.

tracker.py: Handles image processing and Keyframe tracking (GFTT, Optical Flow).

estimator.py: Calculates the position by applying Ridge Regression.

api.py: Generates the JSON formatted output expected by the TEKNOFEST server.

main.py: The entry point to run and test the project.

Usage

You can run the main.py file to execute the code or perform a sample test:

python main.py


For API integration, simply pass your image (frame) read via cv2 to the TeknofestAPIWrapper class.
