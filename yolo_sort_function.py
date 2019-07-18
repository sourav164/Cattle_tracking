# convert output into tensor
import torch
import numpy
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from itertools import cycle
from shapely.affinity import rotate
from shapely.geometry import LineString, Point
from shapely.ops import split

def tensor_out(result):
    data = []
    for i in result :
        list_keys = [ k for k in i.values() ]
        topleft, bottomright = list_keys[2], list_keys[3]
        full = [ float(k) for k in topleft.values()]+[ float(k) for k in bottomright.values()]+[list_keys[1]]+[list_keys[1]]+[1]
        data.append(full)
    output = torch.FloatTensor(data)
    return output

# crop the area of interest
def roi_create(img):
    external_poly = numpy.array( [[[1458,1440],[0,1440],[0,0],[2560,0],[2560,740], [1940,60], [1453,60]]], dtype=numpy.int32 )
    cv2.fillPoly( img , external_poly, (0,0,0) )
    return img

# get the relative width in every cases
def rel_loc(low_x, low_y, high_x, high_y):
    slope, intercept, r_value, p_value, std_err = linregress([low_x,low_y],[high_x,high_y])
    return slope, intercept

# keep track of previous
def keep_retracking(previous_list, previous_id, new_id):
    already_tracked = False
    for lst in previous_list:
        if previous_id in lst:
            lst.append(str(new_id))
            already_tracked = True
    if already_tracked == False:
        previous_list.append([previous_id, str(new_id)])
    return previous_list


def roi_create2(original_frame):
    frame = original_frame.copy()

    # pts - location of the 4 corners of the roi
    pts = numpy.array([[0, 1450], [0, 1087], [977, 80], [1925, 67], [2560, 800], [2560, 1440]])
    (x,y,w,h) = cv2.boundingRect(pts)

    pts = pts - pts.min(axis=0)
    mask = numpy.zeros(original_frame.shape, numpy.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    result = cv2.bitwise_and(original_frame, mask)
    return result

# calculate running mean
def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# take pandas column list and make it a locations running mean
def str_2_xy(str_lst):
    x_loc = [int(i.split(",")[0][1:]) for i in str_lst]
    y_loc = [1440-int(i.split(",")[1][:-1]) for i in str_lst]
    x_loc_mean, y_loc_mean = running_mean(x_loc, 10),  running_mean(y_loc, 10)
    return x_loc_mean, y_loc_mean

# take the dataframe and marge the retracking list
def data_frame_marge(df, retrack_lst):
    for lst in retrack_lst:
        nwl = []
        for i in lst:
            fv = df[i].first_valid_index()
            lv = df[i].last_valid_index()
            vals = df.loc[fv:lv, i].tolist()
            nwl.extend(vals)
            df.drop(i, axis=1, inplace=True)
        df.loc[:, str(lst[0])] = pd.Series(nwl)

    tempt_dic = {}
    for i in df.columns:
        if i != "Unnamed: 0":
            tempt_dic[i] = df[i].count()
    sorted_with_length = sorted(tempt_dic.items(), key=lambda x: x[1], reverse=True)

    return df, sorted_with_length

# cattle direction for retracking, feed x and y after calculating running sum
def cattle_direction(x,y, allowable_dis, point):
    print (x,y, allowable_dis, point)
    x, y = running_mean(x, 3),  running_mean(y, 3)
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    a = (x[0], slope*x[0]+intercept)
    b = (x[-1], slope*x[-1]+intercept)

    # the fitted line
    ab = LineString([a,b])

    # produce circle then arc with 45 degree angle
    circle = Point(b).buffer(allowable_dis)
    left_border = rotate(ab, -135, origin=b)
    right_border = rotate(ab, 135, origin=b)
    splitter = LineString([*left_border.coords, *right_border.coords[::-1]])

    # only get the quarter circle and check if point falls inside it
    sector = split(circle, splitter)[1]
    return sector.contains(Point(point))

def check_accend_decend(x_val_lst, y_val_lst, N):
    x_val = running_mean(x_val_lst, N)
    y_val = running_mean(y_val_lst, N)
    print (x_val, y_val)
    x_val_chk = all(earlier >= later for earlier, later in zip(x_val, x_val[1:])) | all(earlier <= later for earlier, later in zip(x_val, x_val[1:]))
    y_val_chk = all(earlier >= later for earlier, later in zip(y_val, y_val[1:])) | all(earlier <= later for earlier, later in zip(y_val, y_val[1:]))
    return  (len(x_val_lst)>5 ) & (x_val_chk | y_val_chk)
