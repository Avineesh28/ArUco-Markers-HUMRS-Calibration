#!/usr/bin/env python3

import numpy as np
import rosbag
import bisect
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, Float64, Int8
from nav_msgs.msg import Odometry 
from nav_msgs.msg import Path 
from sensor_msgs.msg import JointState 
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose, PoseStamped
import queue
from humrs_msgs.msg import ControlCommand
import rospkg
import sys
import matplotlib.pyplot as plt

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

def interp(arr, times, t, av_fun):
  if t < times[0]:
    print('Time out of range')
    quit()
  elif t > times[-1]:
    print('Time out of range')
    quit()

  step = bisect.bisect_right(times, t) - 1
  if step == len(times) - 1:
    item = np.copy(arr[-1])
  else:
    alpha = (times[step + 1] - t)/(times[step + 1] - times[step])
    item = av_fun(arr[step], arr[step + 1], alpha)

  return item

def zero_order_hold(arr, times, t):
  if t < times[0]:
    print('Time out of range')
    quit()
  elif t > times[-1]:
    print('Time out of range')
    quit()

  step = bisect.bisect_right(times, t) - 1
  return arr[step]

def vanilla_average(x1, x2, gamma):
  return gamma*x1 + (1 - gamma)*x2

def quaternion_average(q1, q2, gamma):
  R1 = R.from_quat(q1)
  R2 = R.from_quat(q2)

  # Partial rotation vector from q1 to q2
  rvec = (R1.inv()*R2).as_rotvec()*gamma

  av_R = R1*R.from_rotvec(rvec)

  return av_R.as_quat()

def odom_average(x1, x2, gamma):
  av = vanilla_average(x1, x2, gamma)
  av[3:7] = quaternion_average(x1[3:7], x2[3:7], gamma)

  return av

def pose_average(x1, x2, gamma):
  av = vanilla_average(x1, x2, gamma)
  av[3:7] = quaternion_average(x1[3:7], x2[3:7], gamma)
  return av

def joint_average(x1, x2, gamma):
  av = vanilla_average(x1, x2, gamma)
  num_joints = len(av)//2
  av[:num_joints] = np.array([angdiff(th1, th2) for th1, th2 in zip(x1[:num_joints], x2[:num_joints])])
  return av

def cmd_values(msg):
  return msg.values

def odom_message_proc(msg):
  p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
  q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
  v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
  w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
  return np.concatenate([p, q, v, w])

def relative_odom_message_proc(msg):
  p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
  q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
  v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
  w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
  frame_w = [msg.frame_angular_velocity.x, msg.frame_angular_velocity.y, msg.frame_angular_velocity.z]
  return np.concatenate([p, q, v, w, frame_w])

def pose_message_proc(msg):
  p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
  q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
  v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z, msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
  return np.concatenate([np.concatenate([p, q]),v])

def joint_message_proc(msg):
  return np.concatenate([msg.position, msg.velocity])

def cmd_message_proc(msg):
  return np.array(msg.values)

def wrench_message_proc(msg):
  f = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
  tau = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
  return np.concatenate([f, tau])

def bool_message_proc(msg):
  return msg.data

bag_file='/home/beau/Documents/humrs_ws/src/humrs_control/bags/trial2.bag'
# cmd_times, cmd_data = process_bag(bag_file, '/humrs/cmd/hmc',cmd_values)#, pose_message_proc)
# odom_times, odom_data = process_bag(bag_file, '/humrs/odom',pose_message_proc)#, joint_message_proc)
ekf_times, ekf_data = process_bag(bag_file, '/humrs/head',pose_message_proc)#, joint_message_proc)
joint_times, joint_data = process_bag(bag_file, '/humrs/fbk/joint_state',joint_message_proc)#, cmd_message_proc)


ekf_time=ekf_times-ekf_times[0]
# odom_time=odom_times-odom_times[0]

titles=['x','y','z','qx','qy','qz','qw','vx','vy','vw','wx','wy','wz']
for i, title in enumerate(titles): 
  plt.subplot(2, 7, i+1)
  plt.plot(ekf_time,ekf_data[:,i],'r')
  # plt.plot(odom_time,odom_data[:,i],'--b')
  plt.title(title)

plt.show()
# i=3
# plt.plot(ekf_time,ekf_data[:,i],'r')
# plt.plot(odom_time,odom_data[:,i],'b')
# plt.show()


# i=0
# plt.plot(ekf_time[1:],(ekf_data[1:,i]-ekf_data[:-1,i])/(ekf_time[1:]-ekf_time[:-1]),'r')
# plt.plot(odom_time,odom_data[:,i+7],'b')
# plt.show()