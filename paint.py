import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import numpy as np
import random

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

all_color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

def get_cycle_pos(all_center, all_radius):
  all_x = []
  all_y = []
  for i in range(0, len(all_center)):
    x = []
    y = []
    center = all_center[i]
    radius = all_radius[i]
    gap = 30
    step = 2*math.pi / gap
    start = 0
    for j in range(0, gap):
      tangle = start + j*step
      x.append(center[0] + radius*math.cos(tangle))
      y.append(center[1] + radius*math.sin(tangle))
    all_x.append(x)
    all_y.append(y)
    
  return all_x, all_y

def dynamic_state_point(attributes, all_labels, all_center, all_radius, data_range, show_train_data):
  fig, ax = plt.subplots()
  x = []
  y = []
  l = ax.plot([],[])
  dot, = ax.plot([],[],marker='o',color='m',markersize=12)
  def init():
    ax.set_xlim(data_range[0][0], data_range[0][1])
    ax.set_ylim(data_range[1][0], data_range[1][1])
    return l,
  
  def gen_dot():
    x_c = (data_range[0][1] + data_range[0][0]) / 2
    y_c = (data_range[1][1] + data_range[1][0]) / 2
    gap = (data_range[0][1] - data_range[0][0]) / 5
    for i in np.linspace(x_c-gap, x_c+gap, 200):
      newdot = [i+random.uniform(0.06,0.09), y_c+random.uniform(0.17,0.26)]
      yield newdot
  def update_dot(newd):
    dot.set_data(newd[0], newd[1])
  
  each_label_x = []
  each_label_y = []
  x_center = []
  y_center = []
  for i in range(0, len(all_center)):
    each_label_x.append([])
    each_label_y.append([])
  for i in range(0, len(attributes)):
    cur_label = all_labels[i]
    cur_sample = attributes[i]
    each_label_x[cur_label].append(cur_sample[0])
    each_label_y[cur_label].append(cur_sample[1])
  all_x_cycle, all_y_cycle = get_cycle_pos(all_center, all_radius)
  for i in range(0, len(all_center)):
    plt.scatter(all_center[i][0], all_center[i][1], s=400, marker='*', color='y')#all_color[i])
    x_cycle = all_x_cycle[i]
    y_cycle = all_y_cycle[i]
    plt.scatter(x_cycle,y_cycle, s=90, marker="+",color=all_color[i])
    # x_center.append(obj[0])
    # y_center.append(obj[1])
  # plt.scatter(x_center,y_center, marker='*',color='red')
  
  for i in range(0, len(each_label_x)):
    cur_x = each_label_x[i]
    cur_y = each_label_y[i]
    if show_train_data:
      plt.scatter(cur_x,cur_y, marker='o', color=all_color[i])
  
  animation = FuncAnimation(fig, update_dot, frames=gen_dot, interval = 200,init_func=init)
  plt.show()

