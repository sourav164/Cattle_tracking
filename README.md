# Cattle Tracking
This program has been designed to work on top of YOLO detection algorithm and SORT tracking algorithm. An assimilation algorithm conssists of multiple user defined function has been used to bridge between the detection and tracking algorithm. To run this code do the following -

1. Install tensorflow implementation of [YOLO](https://github.com/thtrieu/darkflow)
2. Download and place yolo_function.py and SORT.py into YOLO folder
3. Use own training dataset to train the YOLO model
4. Use the trained weight and run the model to track cattle of the pen

# cattle_track.py
Primary file to use other functions. Read images, detect cattle using DL models, deploy SORT for tracking, and use geometric shapes to initiate a lost track.

# yolo_sort_function.py
Functions to initiate lost track

# sort.py
SORT algorithm. Developed by Alex Bewley alex@dynamicdetection.com

# function.py
Previous versions of yolo_sort_function.py. 
