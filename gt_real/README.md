# Creating ground truths using sensors

**NOT IMPLEMENTED** 

Creating path ground truths using GNSS, IMU and visual odometry.

Right now it uses unprocessed live positioning from the device (`livePositionKalman` from `rlog.bz2`). This should be improved by processing GNSS positions with Laika and then applying a kalman filter (probably `LocKalman` from `loc_kf.py`).

# Install

- Install Openpilot with all dependencies using [these instructions](https://github.com/commaai/openpilot/tree/master/tools). Ensure `laika` and Openpilot's `common` and `tools` are then in the Python path.
- `pip install imageio`