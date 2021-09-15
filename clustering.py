import random
import math

# 采用kmean聚类
def get_cluster_center(attributes, k):
  dimension = len(attributes[0])

  # 随机生成聚类中心，范围为[0,1]根据k均分
  first_center_range = []
  range_start = 0.0
  for i in range(0, k):
    end = range_start + 1/k
    first_center_range.append((range_start, end))
    range_start = end

  all_center = []
  for i in range(0, k):
    temp_center = []
    for j in range(0, dimension):
      r = random.uniform(first_center_range[i][0], first_center_range[i][1])
      temp_center.append(r)
    all_center.append(temp_center)
  
  # 循环直至所有聚类中心不再改变
  center_changed = True
  all_labels = []
  for i in range(0, len(attributes)):
    all_labels.append(-1)

  each_label_num = []
  iter_counter = 0
  while center_changed:
    iter_counter+=1
    # print('聚类迭代到第'+str(iter_counter)+'次')
    center_changed = False
    old_center = []
    for i in range(0, k):
      temp_list = []
      cur_center = all_center[i]
      for j in range(0, len(cur_center)):
        temp_list.append(cur_center[j])
      old_center.append(temp_list)
    # 计算当前center下每个样本的标签
    for i in range(0, len(attributes)):
      cur_sample = attributes[i]
      dis_to_all_center = []
      for j in range(0, k):
        dis_to_all_center.append(0)
      for j in range(0, k):
        for l in range(0, dimension):
          dis_to_all_center[j] = dis_to_all_center[j] + (cur_sample[l] - all_center[j][l]) * (cur_sample[l] - all_center[j][l])
      
      min_dis = 999999999
      new_label = -1
      for label_i in range(0, k):
        if dis_to_all_center[label_i] < min_dis:
          new_label = label_i
          min_dis = dis_to_all_center[label_i]
      all_labels[i] = new_label
    
    # 更新center
    each_label_num = []
    for i in range(0, k):
      each_label_num.append(0)
      temp_list = []
      for j in range(0, dimension):
        temp_list.append(0) 
      all_center[i] = temp_list
    for i in range(0, len(attributes)):
      cur_sample = attributes[i]
      cur_label = all_labels[i]
      each_label_num[cur_label] = each_label_num[cur_label] + 1
      for j in range(0, dimension):
        all_center[cur_label][j] += cur_sample[j]

    for i in range(0, k):
      if each_label_num[i] != 0:
        for j in range(0, dimension):
          all_center[i][j] = all_center[i][j] / each_label_num[i]
        if all_center[i] != old_center[i]:
          center_changed = True

      # 如果某个聚类中心一个样本都没吸引到（极有可能出现在一开始），则随机选择一个样本作为聚类中心，并额外迭代一次
      else:
        all_center[i] = attributes[random.randint(0,len(attributes)-1)]
        center_changed = True

  # 获取半径，暂取到该类别下所有样本的距离的均值
  all_radius = []
  for i in range(0, k):
    all_radius.append(0)
  for i in range(0, len(attributes)):
    cur_sample = attributes[i]
    cur_label = all_labels[i]
    cur_center = all_center[cur_label]
    dis = 0
    for j in range(0, dimension):
      dis += (cur_sample[j] - cur_center[j]) * (cur_sample[j] - cur_center[j])
    all_radius[cur_label] += math.sqrt(dis) 
  
  for i in range(0, k):
    if each_label_num[i] != 0:
      all_radius[i] = all_radius[i] / each_label_num[i]
    else:
      all_radius[i] = 0.1

  print('聚类算法迭代了'+str(iter_counter)+'次')
  return all_center, all_labels, all_radius

