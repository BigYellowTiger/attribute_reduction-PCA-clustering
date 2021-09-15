import numpy as np
from numpy.lib.function_base import cov

def get_primary_component(attributes, pc_num):
  # 求每个属性均值
  dimension = len(attributes[0])
  each_att_avg = []
  for i in range(0, dimension):
    each_att_avg.append(0)
  for i in range(0, len(attributes)):
    cur_sample = attributes[i]
    for j in range(0, dimension):
      each_att_avg[j] += cur_sample[j]
  for i in range(0, dimension):
    each_att_avg[i] = each_att_avg[i] / len(attributes)
  # for i in range(0, len(attributes)):
  #   for j in range(0, dimension):
  #     attributes[i][j] = attributes[i][j]-each_att_avg[j]

  # 求协方差矩阵
  cov_matrix = []
  for i in range(0, dimension):
    temp_list = []
    for j in range(0, dimension):
      temp_list.append(0)
    cov_matrix.append(temp_list)
  
  for i in range(0, len(attributes)):
    for dim_i in range(0, dimension):
      for dim_j in range(0, dimension):
        cov_matrix[dim_i][dim_j] += (attributes[i][dim_i] - each_att_avg[dim_i]) * (attributes[i][dim_j] - each_att_avg[dim_j])#attributes[i][dim_i] * attributes[i][dim_j]#

  for i in range(0, dimension):
    for j in range(0, dimension):
      cov_matrix[i][j] = cov_matrix[i][j] / (len(attributes) - 1)
  
  # 求协方差矩阵的特征值和特征向量
  np_cov = np.array(cov_matrix)
  feature_val, feature_vec = np.linalg.eig(np_cov)
  feature_tuple = []
  for i in range(0, len(feature_val)):
    feature_tuple.append((feature_val[i], list(feature_vec[i])))

  # 前pc_num大个特征值的特征向量输出
  feature_tuple = sorted(feature_tuple, key=lambda obj: obj[0], reverse=True)
  result = []
  for i in range(0, pc_num):
    result.append(feature_tuple[i][1])
  
  print('PCA协方差矩阵前n个最大特征值所对应的特征向量：')
  for obj in result:
    print(obj)

  return result
