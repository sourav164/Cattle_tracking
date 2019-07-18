# import all necessary libraries
import cv2, math, numpy, torch, re, time
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import tensorflow as tf
from sort import *
import pandas as pd
from yolo_sort_function import *
import matplotlib.pyplot as plt
from itertools import cycle

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
# input for darknet
options = {'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 3250, 'threshold': 0.35,'gpu': 0.7 }
tfnet = TFNet(options)
slope, intercpet = rel_loc(80,970,1100,2560)

max_cattle =7
lines = ["-","--","-.",":"]
linecycler = cycle(lines)


cv2.namedWindow("Video")

# video input and video write function inputs
cap = cv2.VideoCapture(r'ch08_20181030065128_xvid.avi')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)*65/100
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*65/100
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 29.0, (int(width), int(height)))

totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = [i for i in range (1, totalFrames) if divmod(i, int(2))[1]==0]

# loop through the video, count and resize frames
frames = 0

mot_tracker = Sort()

df = pd.DataFrame()
lst = []
tracked_id = {}
lost_id = []
lost_id_loc = {}
new_id = {}
same_id  = []
retracked_id = []
obj_change_track = []

start = time.time()

# for myFrameNumber in frame_number:
#     cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    disco = {}
    frame = roi_create2(frame)
    frame = cv2.resize(frame,None,fx=0.65,fy=0.65)
    frames += 1
    frame = numpy.asarray(frame)
    det_result = tfnet.return_predict(frame)
    detections = tensor_out(det_result)

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        all_bounding_box = [[(int(x1),int(y1)), (int(x2),int(y1)), (int(x2),int(y2)), (int(x1),int(y2)) ] for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects]
        wrong_identifier = [i for m in all_bounding_box for i in all_bounding_box if ((Polygon(m).contains(Polygon(i))==True) |
         (Polygon(i).area*0.90< Polygon(m).intersection(Polygon(i)).area)) &
         (Polygon(m) != Polygon(i)) & (Polygon(m).area> Polygon(i).area) ]



        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            if ([(int(x1),int(y1)), (int(x2),int(y1)), (int(x2),int(y2)), (int(x1),int(y2))] not in wrong_identifier) and ((x1+x2)/2>900):
                # print (int((x1+x2)/2))

                mean_x, mean_y = (x1+x2)/2 , (y1+y2)/2
                color = colors[int(obj_id) % len(colors)]
                cls = "cow"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame, cls +"-" + str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                disco[str(obj_id)] = int(mean_x), int(mean_y)

                if str(obj_id) not in lst:
                    # only consider maximum 6% deviation from one lost tracking to found
                    allw_dev = int((mean_y*slope+intercpet)*0.06)
                    circle = Point(mean_x,mean_y).buffer(allw_dev)

                    retrack_candid = []

                    for lst_cattle in lost_id:
                        if lst_cattle not in retracked_id:

                            # lost cattles direction using last 30 valid value
                            last_valid_val = [x for x in  df[lst_cattle].values.tolist() if str(x) != 'nan']
                            x_val_lst, y_val_lst = [i for i, j in last_valid_val][-20:], [j for i, j in last_valid_val][-20:]

                            # check if they have accending or decending order

                            lost_x, losty = lost_id_loc[lst_cattle]
                            point = Point (lost_x, losty)
                            if point.within(circle):
                                retrack_candid.append(lst_cattle)
                            elif check_accend_decend(x_val_lst, y_val_lst, 5)==True :
                                try:
                                    if cattle_direction(x_val_lst,y_val_lst, allw_dev*1.67, (lost_x, losty)) == True:
                                        retrack_candid.append(lst_cattle)
                                except:
                                    pass

                    if len(retrack_candid)>1:
                        retrack_candid_length = [int(df[[lst_cattle]].count()[0]) for lst_cattle in retrack_candid]
                        retrack_result = retrack_candid[retrack_candid_length.index(max(retrack_candid_length))]

                        # print ("retrack", frames, obj_id, retrack_result)
                        keep_retracking(obj_change_track, retrack_result, obj_id)
                        retracked_id.append(retrack_result)


                    elif len(retrack_candid) ==1:
                        # print ("retrack", frames, obj_id, retrack_candid[0])
                        keep_retracking(obj_change_track, retrack_candid[0], obj_id)
                        retracked_id.append(retrack_candid[0])


                    lst.append(str(obj_id))

                else :
                	# add the last valid location to dictionary
                	tracked_id[str(obj_id)] = (frames, mean_x, mean_y)

    df = df.append(disco , ignore_index=True)
    if frames>30:
        for i in lst:
            if i not in lost_id:
                m =  df[str(i)][-5:]
                sin_col = df[[i]]


                # print (m)
                if (m.isnull().all()) == True and i not in lost_id:
                    # index of last non-zero value
                    sin_col = df[[i]]
                    nz_ind = sin_col.apply(pd.Series.last_valid_index)[0]
                    # x,y = [int(re.sub("\D", "",i)) for i in sin_col.iloc[nz_ind][0].split(",")]
                    x,y = sin_col.iloc[nz_ind][0]
                    lost_id.append(i)
                    lost_id_loc[str(i)] = int(x), int(y)


    cv2.imshow('Video', frame)
    out.write(frame)
    # print (frames)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()

df.to_csv("data2.csv")

df = pd.read_csv("data2.csv")
df, sorted_with_length = data_frame_marge(df, obj_change_track)

print (sorted_with_length)

for i in range(0, max_cattle):
    # convert the column into a list
    data_list = [x for x in list(df[sorted_with_length[i][0]]) if str(x) != 'nan']
    x_loc_mean, y_loc_mean = str_2_xy(data_list)
    plt.plot(x_loc_mean, y_loc_mean, next(linecycler), label = sorted_with_length[i][0])
plt.legend(title='Cattle Tracking')
plt.show()



end = time.time()
time_taken = end - start
print('Time: ',time_taken)
print (obj_change_track)
