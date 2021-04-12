
# Tools
Available scripts:

- rosbag_to_h5.py: Convert `rosbag` datasets into `h5`. For example:

```
python rosbag_to_h5.py ECD/rosbag/ --output_dir ECD/hdf5/ --image_topic /dvs/image_raw
```

- random_crop.py: Crop an `h5` sequence into small segments of a certain duration and spatial resolution. Handy for creating training datasets. For example:

```
python3 random_crop.py datasets/UZH-FPV/ --output_dir B2EB/ --original_res "(260,346)" --with_
images True
```
# DVS Datasets

**TODO**: Store these dataset in some local server.

### Event-Camera Dataset (ECD)

  - [Original link](http://rpg.ifi.uzh.ch/davis_data.html)

 - Associated paper:
*Mueggler, E., Rebecq, H., Gallego, G., Delbruck, T., & Scaramuzza, D. (2017). The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International Journal of Robotics Research, 36 (2), 142-149.*
[PDF](https://arxiv.org/pdf/1610.08336.pdf)

- Sensor: DAVIS 240C (240x180)

- Format: rosbag

```
/optitrack/davis --- geometry_msgs_PoseStamped
/dvs/camera_info --- sensor_msgs_CameraInfo
/dvs/imu --- sensor_msgs_Imu
/dvs/events --- dvs_msgs_EventArray
/dvs/image_raw --- sensor_msgs_Image
```

Note: To remove redundant data, I only collected some of the sequences of this dataset.

### High Quality Frames (HQF) Dataset

  - [Original link](https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_)

 - Associated paper:
*Stoffregen, T., Scheerlinck, C., Scaramuzza, D., Drummond, T., Barnes, N., Kleeman, L., & Mahony, R. (2020). Reducing the Sim-to-Real Gap for Event Cameras. In European Conf. Comput. Vis. (ECCV).*
[PDF](https://arxiv.org/pdf/2003.09078.pdf)

- Sensor: DAVIS 240C (240x180)

- Format: rosbag

```
/dvs/events --- dvs_msgs_EventArray
/dvs/image_raw --- sensor_msgs_Image
```

### UZH-FPV Drone Racing Dataset

 - [Original link](https://fpv.ifi.uzh.ch/?page_id=50)

 - Associated paper:
*Delmerico, J., Cieslewski, T., Rebecq, H., Faessler, M., & Scaramuzza, D. (2019, May). Are we ready for autonomous drone racing? the UZH-FPV drone racing dataset. In 2019 International Conference on Robotics and Automation (ICRA) (pp. 6713-6719). IEEE.*
[PDF](http://rpg.ifi.uzh.ch/docs/ICRA19_Delmerico.pdf)

- Sensor: mDAVIS 346 (346x260)

- Format: rosbag
```
/dvs/imu --- sensor_msgs_Imu
/dvs/events --- dvs_msgs_EventArray
/groundtruth/pose --- geometry_msgs_PoseStamped
/dvs/image_raw --- sensor_msgs_Image
```

Note: Only indoor forward facing sequences.
