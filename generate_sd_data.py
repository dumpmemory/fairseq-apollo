import numpy as np
import sys
import os


folder_path = sys.argv[1]
def generate(file_path, num):
    sampled_data = np.random.randint(low=1, high=128, size=(num, 511))
    zero_data = np.zeros((num, 1), dtype='int32')
    zero_sampled_data = np.concatenate((zero_data, sampled_data), axis=1)
    final_data = np.concatenate((zero_sampled_data, zero_sampled_data), axis=1)
    np.savetxt(file_path, final_data, fmt='%d')

file_path = os.path.join(folder_path, 'train' + '.txt')
generate(file_path, 100000)
file_path = os.path.join(folder_path, 'val' + '.txt')
generate(file_path, 10000)
file_path = os.path.join(folder_path, 'test' + '.txt')
generate(file_path, 10000)


