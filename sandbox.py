
#%%
# setup
#base
import os
import numpy as np
import pandas as pd

import json
import time 

import random

#tensorflow

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Subtract, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils import plotting as plt # TODO rename shortcut
from utils import pretrain_task as pretrain



#%%

training_path = os.getcwd()+'/data/training/'
train_data = io.get_tasks(training_path)

class logger:
    def __init__(self, ID):
        self.name = ID
        self.loss = {}
        self.model = None
        self.model_path = 'data/weights/{}.h5'.format(ID)
        self.logs_path = 'data/logs/{}.json'.format(ID)

    def save_experiment(self, model, history):
        self.model = model 
        
        # save NN
        model.save(self.model_path)

        # save loss
        loss = history.history
        for l in loss:
            loss[l] = [float(num) for num in loss[l]]
        self.loss = loss
        print(os.getcwd())
        with open(self.logs_path, 'w') as json_file:
            json.dump(self.loss, json_file)

    def load_experiment(self):
        # load model
        model = load_model(self.model_path)

        # load loss
        with open(self.logs_path, 'r') as json_file:
            loss = json.load(json_file)
        log.loss = loss
        return model


def enhance_mat_30x30(mat):
    empty_array = np.full((30, 30), 0, dtype=np.float32)
    if(len(mat) != 0):
        mat = np.asarray(mat, dtype=np.float32)
        empty_array[:mat.shape[0], : mat.shape[1]] = mat
        
    return np.expand_dims(empty_array, axis= 2) 

#%%
# load experiment 


log = logger('three_input_feature_extraction_model')
model = log.load_experiment()
model.summary()


#%%

from pandas.api.types import CategoricalDtype
cat_color = CategoricalDtype(categories=[i for i in range(10)]) 

flatten = lambda x: [item for row in x for item in row]
train_input = []
train_output = []

train_input_2 = []
train_output_2 = []
y_train_col_len  = []
y_train_row_len  = []
y_train_unique_colors_sum = []
y_train_unique_colors_cat = []
test_input_list = []
test_output_list = []

for task in train_data:
    for sample in task['train']:

        input_1 = sample['input']
        sample_2 = {'input':[]}
        while sample_2['input'] != input_1:
            sample_2 = random.choice(task['train'])
        
        train_input.append(enhance_mat_30x30(input_1))
        train_output.append(enhance_mat_30x30(sample['output']))
        
        train_input_2.append(enhance_mat_30x30(sample_2['input']))
        train_output_2.append(enhance_mat_30x30(sample_2['output']))

        unique_colors = pd.Series(list(set(flatten(sample['output'])))).astype(cat_color)
        y_train_col_len.append(len(sample['output']))
        y_train_row_len.append(len(sample['output'][0]))
        y_train_unique_colors_sum.append(len(set(flatten(sample['output']))))
        y_train_unique_colors_cat.append([i in unique_colors for i in range(10)])

    for sample in task['test']:
        test_input_list.append(enhance_mat_30x30(sample['input']))
        test_output_list.append(enhance_mat_30x30(sample['output']))

#%%
# Raphael
# Prepare data for pretrain tasks

"""
The model will learn to predict which transformations happend to the input task.

ONE !11 transformation will be applied at the task at each time.

Input1 - task_1_input
Input2 - task_2_input
Input3 - task_2_output

TODO:
Would this be better?
    Input1
    Output1
    Input2
    Output2

Output1 - output1_row
Output2 - output1_col
TODO: >> Output3 - unique_col_sum (will use this from input_1)
TODO: >> Output4 - unique_col_cat (will use this from input_1)
Output5 - rotation_angle
Output6 - row or column removed
Output7 - line number of removed
Output8 - line or column shifted
Output9 - line number of shifted

"""

train_input_1 = []
train_input_2 = []
train_output_2 = []

y_train_col_len  = []
y_train_row_len  = []
y_train_unique_colors_sum = []
y_train_unique_colors_cat = []
y_train_rotation_angle = []
y_train_line_or_column_removed = []
y_train_line_nr_removed = []
y_train_line_or_column_shifted = []
y_train_line_nr_shifted = []

cnt = 0

for task in train_data:
    for sample in task['train']:
        input_1 = sample['input']

        sample_2 = {'input':[]}
        while sample_2['input'] != input_1:
            sample_2 = random.choice(task['train'])
            
        input_2 = sample_2['input']
        
        if len(input_1) < 2 or len(input_2) < 2 or len(input_1[0]) < 2 or len(input_2[0]) < 2:
            cnt = cnt + 1
            continue
        
        train_input_1.append(enhance_mat_30x30(input_1))
        
        y_train_unique_colors_sum.append(len(set(flatten(sample['input']))))
        y_train_unique_colors_cat.append([i in unique_colors for i in range(10)])
        
        # do not enhance bc. transformation need to be applied
        train_input_2.append(input_2)
        
        
print("Counter: ")
print(cnt)
y_rotation_task_list, y_rotation_angle =  pretrain.rotate_tasks(train_input_2)

y_line_shifted, y_row_or_column_shifted, y_line_nr_shifted = pretrain.shift_line_tasks(train_input_2)

y_line_removed, y_row_or_column_removed, y_line_nr_removed = pretrain.remove_line_tasks(train_input_2)

y_multiplied_tasks, y_multiply = pretrain.multiply_tasks(train_input_2)

# No I want to do some one_hot_encode stuff
# Transformation1 | Transformation2 | Transformation 3 ...
#               0 |               1 |                0

# 1. Copy train_input_1 3 times in this list
# 2. Copy train_input_2 3 times in this list
# 3. Y_rotation_angle = fill up with 0 accordingly
# 4. y_row_or_column_removed = fill up accordingly 
# 5. y_train_line_nr_removed = fill up accordingly 
# 6. y_row_or_column_shifted = fill up accordingly 
# 7. y_train_line_nr_shifted = fill up accordingly 
# 8. y_multiply


# Print lens to be safe
print(len(train_input_1))
print(len(train_input_2))
print(len(y_rotation_task_list))
print(len(y_rotation_angle))
print(len(y_line_shifted))
print(len(y_row_or_column_shifted))
print(len(y_line_nr_shifted))

print(len(y_line_removed))
print(len(y_row_or_column_removed))
print(len(y_line_nr_removed))

print(len(y_train_unique_colors_sum))
print(len(y_train_unique_colors_cat))

# used for pretrain tasks except multiply
zeros = [0] * 1214

# used for multiply
# if no multiply is applied size factor = 1
ones = [1] * 1214

print(len(zeros))

# TODO refactor this to a method in a way
# that an arbitrary number of pretrain tasks can be easily applied

# 0)
train_input_2 = [enhance_mat_30x30(el) for el in train_input_2]

# 1) 2)
train_input_1 = train_input_1 + train_input_1 + train_input_1 + train_input_1
train_input_2 = train_input_2 + train_input_2 + train_input_2 + train_input_2 

# 2)
train_output_2 = [enhance_mat_30x30(el) for el in y_rotation_task_list]
train_output_2 = train_output_2 + [enhance_mat_30x30(el) for el in y_line_shifted]
train_output_2 = train_output_2 + [enhance_mat_30x30(el) for el in y_line_removed]
train_output_2 = train_output_2 + [enhance_mat_30x30(el) for el in y_multiplied_tasks]

# 3)
y_rotation_angle = y_rotation_angle + zeros + zeros + zeros

# 4)
y_row_or_column_shifted = zeros + y_row_or_column_shifted + zeros + zeros
y_line_nr_shifted = zeros + y_line_nr_shifted + zeros + zeros
y_train_line_or_column_removed = zeros + y_train_line_or_column_removed + zeros + zeros

# 5)
y_row_or_column_removed = zeros + zeros + y_row_or_column_removed + zeros
y_line_nr_removed = zeros + zeros + y_line_nr_removed + zeros

# 6)
y_multiply = ones + ones + ones + y_multiply 


#%%

# input
input_X1 = Input(shape=(30, 30, 1))
input_X2 = Input(shape=(30, 30, 1))
output_X2 = Input(shape=(30, 30, 1))

# convolution layers
x_1 = Conv2D(64, (3, 3), activation='relu')(input_X1)
x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
x_1 = Dropout(0.25)(x_1)
x_1 = Flatten()(x_1)

x_2 = Conv2D(64, (3, 3), activation='relu')(input_X2)
x_2 = MaxPooling2D(pool_size=(2, 2))(x_2)
x_2 = Dropout(0.25)(x_2)
x_2 = Flatten()(x_2)

x_2_out = Conv2D(64, (3, 3), activation='relu')(output_X2)
x_2_out = MaxPooling2D(pool_size=(2, 2))(x_2_out)
x_2_out = Dropout(0.25)(x_2_out)
x_2_out = Flatten()(x_2_out)

merge = concatenate([x_1, x_2, x_2_out])

merge = Dense(128, activation='relu')(merge)
merge = Dropout(0.5)(merge)

# regression layers
out_1 = Dense(128, activation='relu')(merge)
out_1 = Dense(1, activation='linear', name='rows')(out_1)

out_2 = Dense(128, activation='relu')(merge)
out_2 = Dense(1, activation='linear', name='cols')(out_2)

out_3 = Dense(128, activation='relu')(merge)
out_3 = Dense(1, activation='linear', name='unique_colors_sum')(out_3)

# multi-label classification layers
out_4 = Dense(128, activation='relu')(merge)
out_4 = Dense(10, activation='sigmoid', name='unique_colors_cat')(out_4)

model = Model(inputs=[input_X1, input_X2, output_X2], outputs=[out_1, out_2, out_3, out_4])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "rows": "mean_absolute_error",
    "cols": "mean_absolute_error",
    "unique_colors_sum": "mean_absolute_error",
    "unique_colors_cat": "binary_crossentropy"
    }

model.compile(loss=losses, optimizer=opt)



#%%

start = time.time()

history = model.fit(
    [
        np.array(train_input),
        np.array(train_input_2),
        np.array(train_output_2)],
    [
        np.array(y_train_col_len),
        np.array(y_train_row_len),
        np.array(y_train_unique_colors_sum),
        np.array(y_train_unique_colors_cat)
        ],
    epochs=500)

print('training time {} minutes'.format(round(time.time()-start)/60))


#%%

log = logger('three_input_feature_extraction_model')
log.save_experiment(model, history)


#%%

plt.plot_loss(log, 'rows_loss', save=False)


#%%


#%%

y_hat = model.predict([
        np.array(train_input),
        np.array(train_input_2),
        np.array(train_output_2)])
y_hat[0][0]


#%%

y_hat[0][1]

#%%

y_train_row_len[1]

# %%
# plot input output

plt.plot_matrix(train_data[0]['train'][1]['input'])
plt.plot_matrix(train_data[0]['train'][1]['output'])

