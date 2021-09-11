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

fig, ax = plt.subplots(2,2)
ax[0][0].set_title('原始数据直接PCA后样本分布')
ax[0][1].set_title('属性约简+PCA后样本分布')
ax[1][0].set_title('属性约简+PCA+聚类挖掘结果')
ax[1][1].set_title('各个模式中心点与边界分布')
def dynamic_state_point(attributes, all_labels, all_center, all_radius, data_range, show_train_data, label_num, real_labels, ori_attributes):
  # 数据原始分布
  all_x = []
  all_y = []
  for i in range(0, label_num):
    all_x.append([])
    all_y.append([])
  for i in range(0, len(ori_attributes)):
    cur_sample = ori_attributes[i]
    cur_label = real_labels[i]
    all_x[cur_label-1].append(cur_sample[0])
    all_y[cur_label-1].append(cur_sample[1])
  for i in range(0, label_num):
    cur_x = all_x[i]
    cur_y = all_y[i]
    ax[0][0].scatter(cur_x, cur_y, s = 60, color = all_color[i])
  
  # 约简后原始数据集分布
  all_x = []
  all_y = []
  for i in range(0, label_num):
    all_x.append([])
    all_y.append([])
  for i in range(0, len(attributes)):
    cur_sample = attributes[i]
    cur_label = real_labels[i]
    all_x[cur_label-1].append(cur_sample[0])
    all_y[cur_label-1].append(cur_sample[1])
  for i in range(0, label_num):
    cur_x = all_x[i]
    cur_y = all_y[i]
    ax[0][1].scatter(cur_x, cur_y, s = 60, color = all_color[i])
  
  # 约简+pca+聚类后分布
  each_label_x = []
  each_label_y = []
  for i in range(0, len(all_center)):
    each_label_x.append([])
    each_label_y.append([])
  for i in range(0, len(attributes)):
    cur_label = all_labels[i]
    cur_sample = attributes[i]
    each_label_x[cur_label].append(cur_sample[0])
    each_label_y[cur_label].append(cur_sample[1])
  
  for i in range(0, len(each_label_x)):
    cur_x = each_label_x[i]
    cur_y = each_label_y[i]
    if show_train_data:
      ax[1][0].scatter(cur_x,cur_y, marker='o', color=all_color[i])

  # 约简+pca+聚类后分布
  l = ax[1][1].plot([],[])
  dot, = ax[1][1].plot([],[],marker='o',color='m',markersize=12)
  def init():
    ax[1][1].set_xlim(data_range[0][0], data_range[0][1])
    ax[1][1].set_ylim(data_range[1][0], data_range[1][1])
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

  all_x_cycle, all_y_cycle = get_cycle_pos(all_center, all_radius)
  for i in range(0, len(all_center)):
    ax[1][1].scatter(all_center[i][0], all_center[i][1], s=400, marker='*', color='y')#all_color[i])
    x_cycle = all_x_cycle[i]
    y_cycle = all_y_cycle[i]
    ax[1][1].scatter(x_cycle,y_cycle, s=90, marker="+",color=all_color[i])
  
  animation = FuncAnimation(fig, update_dot, frames=gen_dot, interval = 200,init_func=init)
  plt.show()