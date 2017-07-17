# roomba-mapping
A SLAM algorithm for roomba robots using an Extended Kalman Filter.

The logfile viewer for visualization of the map is taken from Claus Brenner's course on SLAM: https://www.youtube.com/playlist?list=PLpUPoM7Rgzi_7YWn14Va2FODh7LzADBSm

The data collector (robot_data_collector.py) script records a stream of data from the robot as it runs, and the main algorithm (extended_kalman_slam.py) attempts to use this data to map the robot's path and obstacles it encounters.


