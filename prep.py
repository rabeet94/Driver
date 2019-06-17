# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:20:48 2019

@author: DELL
"""

import pandas as pd
import glob

#
def create_rawframe(raw_path):
    path = raw_path
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    
    return frame

#
def create_labelframe(label_path):
    path = label_path
    labels = create_rawframe(raw_path = path)
    labelcounts = labels['bookingID'].value_counts()
    to_remove = labelcounts[labelcounts == 1].index
    labels = labels[labels.bookingID.isin(to_remove)]

    return labels

#
def create_mergedframe(raw_path, label_path = None):
    rawdata = create_rawframe(raw_path = raw_path)
    if label_path is not None:
        labels = create_labelframe(label_path = label_path)
        rawdata = rawdata.merge(labels, on = 'bookingID')
        
    return rawdata

#used for creating features out of initial data
def create_featureframe(raw_path, label_path, labels = True):
    merged = create_mergedframe(raw_path = raw_path, label_path = label_path)
    bidlist = merged['bookingID'].unique()
    bidframelist = []
    headers = ['maxspeed', 'meanspeed', 'maxseconds', 'overscounts',
               'accxmax', 'accxmin', 'accymax', 'accymin', 'acczmax', 'acczmin',
               'gyroxmax', 'gyroxmin', 'gyroymax', 'gyroymin', 'gyrozmax', 'gyrozmin',
               'accxmean', 'accymean', 'acczmean',
               'gyroxmean', 'gyroymean', 'gyrozmean',
               'haccxcounts', 'haccycounts', 'hacczcounts',
               'hdccxcounts', 'hdccycounts', 'hdcczcounts',
               'hcgyxcounts', 'hcgyycounts', 'hcgyzcounts',
               'gyroxstd', 'gyroystd', 'gyrozstd'
               ]
    if labels == True:
        headers.append('label')
        
    haccxl, haccyl, hacczl = 5, 5, 5
    hdccxl, hdccyl, hdcczl = -5, -5, -5
    hcgyxl, hcgyyl, hcgyzl = 1, 1, 1
    count = 0
    for bid in bidlist:
        idframe = merged[merged['bookingID'] == bid].sort_values(by = 'second')
        didframe = idframe.describe()
        variables = [didframe.loc['max','Speed'], didframe.loc['mean', 'Speed'], didframe.loc['max', 'second'], len(idframe[idframe['Speed'] > 15]),
                     didframe.loc['max', 'acceleration_x'], didframe.loc['min', 'acceleration_x'], didframe.loc['max', 'acceleration_y'], didframe.loc['min', 'acceleration_y'], didframe.loc['max', 'acceleration_z'], didframe.loc['min', 'acceleration_z'],
                     didframe.loc['max', 'gyro_x'], didframe.loc['min', 'gyro_x'], didframe.loc['max', 'gyro_y'], didframe.loc['min', 'gyro_y'], didframe.loc['max', 'gyro_z'], didframe.loc['min', 'gyro_z'],
                     didframe.loc['mean', 'acceleration_x'], didframe.loc['mean', 'acceleration_y'], didframe.loc['mean', 'acceleration_z'],
                     didframe.loc['mean', 'gyro_x'], didframe.loc['mean', 'gyro_y'], didframe.loc['mean', 'gyro_z'],
                     len(idframe[idframe['acceleration_x'] > haccxl]), len(idframe[idframe['acceleration_y'] > haccyl]), len(idframe[idframe['acceleration_z'] > hacczl]),
                     len(idframe[idframe['acceleration_x'] < hdccxl]), len(idframe[idframe['acceleration_y'] < hdccyl]), len(idframe[idframe['acceleration_z'] < hdcczl]),
                     len(idframe[(idframe['gyro_x'] > hcgyxl) | (idframe['gyro_x'] < -(hcgyxl))]), len(idframe[(idframe['gyro_y'] > hcgyyl) | (idframe['gyro_y'] < -(hcgyyl))]), len(idframe[(idframe['gyro_z'] > hcgyzl) | (idframe['gyro_z'] < -(hcgyzl))]),
                     didframe.loc['std', 'gyro_x'], didframe.loc['std', 'gyro_y'], didframe.loc['std', 'gyro_z']
                     ]
        if labels == True:
            variables.append(idframe.reset_index().loc[0, 'label'])
        bidframelist.append(variables)
        if count % 100 == 0:
            print(count)
        count+=1

    fframe = pd.DataFrame(bidframelist, columns = headers)
    fframe = fframe[fframe['meanspeed'] > 0]
    fframe['overscounts'] = fframe['overscounts']/fframe['maxseconds']
    fframe['haccxcounts'] = fframe['haccxcounts']/fframe['maxseconds']
    fframe['haccycounts'] = fframe['haccycounts']/fframe['maxseconds']
    fframe['hacczcounts'] = fframe['hacczcounts']/fframe['maxseconds']
    fframe['hdccxcounts'] = fframe['hdccxcounts']/fframe['maxseconds']
    fframe['hdccycounts'] = fframe['hdccycounts']/fframe['maxseconds']
    fframe['hdcczcounts'] = fframe['hdcczcounts']/fframe['maxseconds']
    
    fframe['hcgyxcounts'] = fframe['hcgyxcounts']/fframe['maxseconds']
    fframe['hcgyycounts'] = fframe['hcgyycounts']/fframe['maxseconds']
    fframe['hcgyzcounts'] = fframe['hcgyzcounts']/fframe['maxseconds']
    
    return fframe