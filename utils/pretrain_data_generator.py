"""
This file will be used to generate data for the pretrain tasks.

Interface:
==========
def generate_pretrain_data(pretrain_functions, task_list)

Inputs:
==========
It will take two lists:
1. list of pretrain functions -> see pretrain_task.py
2. list of tasks which will be used for pretrain (originals)

Returns:
==========
1. list of modified tasks
2. list of y_label lists
"""

import utils.pretrain_task as pretrain
import numpy as np


def generate_pretrain_data(pretrain_functions, task_list):
    tmp_result = [pretrain_fun(task_list) for pretrain_fun in pretrain_functions]
    zeros = [0] * len(task_list)
    
    modified_tasks = []
    
    # this list will contain all the y_label list of the pretrain tasks
    y_labels_list = []
    
    for counter, pretrain_result in enumerate(tmp_result):
        enhanced_pretrain_tasks = [enhance_mat_30x30(task) for task in pretrain_result[0]]
        
        modified_tasks.extend(enhanced_pretrain_tasks)
        
        for i in range(1, len(pretrain_result)):
            y_labels = pretrain_result[i]
            for j in range(counter):
                y_labels = zeros + y_labels
            
            for j in range(counter, len(tmp_result) - 1):
                y_labels = y_labels + zeros
        
            y_labels_list.append(y_labels)
    
    return modified_tasks, y_labels_list


def enhance_mat_30x30(mat):
    empty_array = np.full((30, 30), 0, dtype=np.float32)
    if(len(mat) != 0):
        mat = np.asarray(mat, dtype=np.float32)
        empty_array[:mat.shape[0], : mat.shape[1]] = mat
        
    return np.expand_dims(empty_array, axis= 2) 


if __name__ == "__main__":
    pretrain_functions = [pretrain.rotate_tasks, pretrain.multiply_tasks]
    
    test_task_1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    test_task_2 = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    test_task_3 = [[1, 2, 3], [2, 2, 3], [3, 3, 3]]
    
    test_task_list = [test_task_1, test_task_2, test_task_3]
    
    modified_tasks, y_label_list = generate_pretrain_data(pretrain_functions, test_task_list)
    
    print("Modified_tasks: ")
    # print(modified_tasks)
    
    print("Len modified_tasks: ")
    print(len(modified_tasks))
    
    print("y_label_list: ")
    print(y_label_list)
    
    print("Len y_label_list: ")
    print(len(y_label_list))
    
    print("Len y_label_list[0]: ")
    print(len(y_label_list[0]))
    
    print("Len y_label_list[1]: ")
    print(len(y_label_list[1]))
    
    

    # to verify nothing in this list changed
    print(test_task_list)
    assert test_task_list == [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[3, 3, 3], [2, 2, 2], [1, 1, 1]], [[1, 2, 3], [2, 2, 3], [3, 3, 3]]]