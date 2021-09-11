from attribute_reduction import bba_attribute_reduction
import copy
import clustering
import pca
import paint
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'C:/Users/qly/Desktop/磨矿系统需求/蚯蚓图demo/train/wine.csv'
data_f = open(file_path,'r') # 默认csv文件格式，且经过标准化
attributes = []
real_labels = []
for obj in data_f:
  temp_arr = obj.split(',')
  temp_list = []
  for i in range(0, len(temp_arr)-1):
    temp_list.append(float(temp_arr[i]))
  attributes.append(temp_list)
  real_labels.append(int(temp_arr[len(temp_arr)-1]))
ori_attributes = copy.deepcopy(attributes)

# 属性约简
reduct = bba_attribute_reduction(file_path)
for i in range(0, len(attributes)):
  cur_sample = attributes[i]
  temp_list = []
  for j in range(0, len(reduct)):
    if reduct[j] == 1:
      temp_list.append(cur_sample[j])
  attributes[i] = temp_list

# pca 降维
pc = pca.get_primary_component(attributes, 2)
np_pc = np.array(pc)
np_attributes = np.array(attributes)
np_reduced_attributes = np.dot(np_attributes, np_pc.T)
attributes = np_reduced_attributes.tolist()

# 对原始数据集直接pca降维
pc = pca.get_primary_component(ori_attributes, 2)
np_pc = np.array(pc)
np_ori_attributes = np.array(ori_attributes)
np_reduced_ori_attributes = np.dot(np_ori_attributes, np_pc.T)
ori_attributes = np_reduced_ori_attributes.tolist()

# 聚类
all_center, all_labels, all_radius = clustering.get_cluster_center(attributes, 3)

# 获取绘图范围
x_max = -10000000
x_min = 10000000
y_max = -10000000
y_min = 10000000
for i in range(0, len(attributes)):
  cur_sample = attributes[i]
  if cur_sample[0] > x_max:
    x_max = cur_sample[0]
  if cur_sample[0] < x_min:
    x_min = cur_sample[0]
  if cur_sample[1] > y_max:
    y_max = cur_sample[1]
  if cur_sample[1] < y_min:
    y_min = cur_sample[1]
data_range = [[x_min, x_max], [y_min, y_max]]
# demo动画
paint.dynamic_state_point(attributes, all_labels, all_center, all_radius, data_range, True, 3, real_labels, ori_attributes)