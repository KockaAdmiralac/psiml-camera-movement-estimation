# Camera Movement Estimation
A PSI:ML 6 project.

## Installation
To install project's required packages:
```console
$ pip install --user -r requirements.txt
```

Running the project requires the following project structure:
```
├── code
├─┬ dataset
│ ├─ poses
│ └─ sequences
```
The files inside the `dataset` folder can be downloaded from [KITTI's website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images, calibration data and ground truth poses). **Note:** You may need to replace calib.txt files inside sequences with calib.txt files from calibration data (merge sequence folders from grayscale and calibration data, with calibration files having precedence).
