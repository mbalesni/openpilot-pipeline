# Path ground-truth creation

Creating path ground truth positions for training the supercombo model.

Right now it uses unprocessed live positioning (`livePositionKalman` from `rlog.bz2`). This will later be improved by processing GNSS positions with Laika and then applying a more complex kalman filter (probably `LocKalman` from `loc_kf.py`).

# Install

- Install Openpilot with all dependencies using [these instructions](https://github.com/commaai/openpilot/tree/master/tools). Ensure `laika` and Openpilot's `common` and `tools` are then in the Python path.
- `pip install imageio`