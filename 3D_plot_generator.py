#!/usr/bin/env python3

import numpy as np
import rosbag
import math
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Gets sorted time array and corresponding data array, where data
# is created by applying msg_proc to messages
def process_bag(bag_file, topic, msg_proc):
  times = []
  data = []
  for topic, msg, t in rosbag.Bag(bag_file).read_messages(topics=[topic]):
    times.append(t.to_sec())
    # data.append(msg.values)
    data.append(msg_proc(msg))

  times = np.array(times)
  data = np.array(data)

  sort_idx = np.argsort(times)

  times = times[sort_idx]
  data = data[sort_idx]

  return times, data

def Tpose_message_proc(msg):
    data = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    return data

def get_angle_and_graph(data_x,data_y,data_z,time):
  ax = plt.axes(projection="3d")
  cmap=plt.cm.viridis
  norm = plt.Normalize(min(time),max(time))
  colors = cmap(norm(time))
  ax.scatter(data_x, data_y, data_z, c=colors, cmap=cmap, norm=norm)
  ax.plot(data_x, data_y, data_z, color='black')
  ax.set_title("Change in X_pose, Y_pose, Z_pose of detected Tag")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")
  # plt.colorbar(label='Time', cmap=cmap, norm=norm)

  # plt.plot(time, data_x)
  # plt.show()
  max_index= data_z.index(max(data_z))
  min_index= data_z.index(min(data_z))

  point_a=[data_x[min_index], data_y[min_index], data_z[min_index]]
  point_b=[data_x[max_index], data_y[max_index], data_z[max_index]]
  point_c=[data_x[min_index], data_y[min_index], data_z[max_index]]

  vector_AB= [xi-yi for xi,yi in zip(point_b,point_a)]
  vector_AC= [xi-yi for xi,yi in zip(point_c,point_a)]

  mag_AB=math.sqrt(sum(pow(element, 2) for element in vector_AB))
  mag_AC=math.sqrt(sum(pow(element, 2) for element in vector_AC))

  theta = math.acos(np.dot(vector_AB,vector_AC)/(mag_AB*mag_AC))
  print(f"Radians: {theta}")
  print(f"Degrees: {math.degrees(theta)}")
  # Visualization
  # plot_arrows_and_angle(point_a, point_b, point_c, theta)

  print(f"Start: ",(min(data_z)))
  print(f"Peak: ",(max(data_z)))
  plt.show()

savefolder="/home/biorobotics/Apriltags_test/2024-03-26-Afternoon/"
bag_folder='/home/biorobotics/stable_ws/src/humrs_ros/humrs_vrep/humrs_control/bags/'
timestamp='2024-03-26-15-57-46'
bag_prefix='april_on_land__'
bag_file=f"{bag_folder}{bag_prefix}{timestamp}.bag"
Tpose_times, Tpose_data = process_bag(bag_file, '/humrs/cam', Tpose_message_proc)
data_x=[]
data_y=[]
data_z=[]
time=Tpose_times

for i in Tpose_data:
    data_x.append(i[0])
    data_y.append(i[1])
    data_z.append(i[2])

# for Entire data graph
# get_angle_and_graph(data_x,data_y,data_z)

# For Interval graph
# n=len(data_x)
# # Cleaning part where IU is calibrated
# data_x=data_x[int(n/2):]
# data_x=data_y[int(n/2):]
# data_x=data_z[int(n/2):]

n=len(data_x)
for i in range(1,9):
  x=int(((i-1)*n)/8)
  y=int((i*n)/8)
  print(f"======  Quarter ",i,"  ======")
  get_angle_and_graph(data_x[x:y+1],data_y[x:y+1],data_z[x:y+1],time[x:y+1])
    
