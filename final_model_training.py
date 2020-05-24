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
from tensorflow.keras.estimator import model_to_estimator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils.logging import Logger
from utils import plotting as plt # TODO rename shortcut


#%%

training_path = os.getcwd()+'/data/training/'
train_data = io.get_tasks(training_path)


def enhance_mat_30x30(mat):
    empty_array = np.full((30, 30), 0, dtype=np.float32)
    if(len(mat) != 0):
        mat = np.asarray(mat, dtype=np.float32)
        empty_array[:mat.shape[0], : mat.shape[1]] = mat
        
    return np.expand_dims(empty_array, axis= 2) 

#%%

log = Logger('2_input_12_pretrain')
model = log.load_experiment()
model.summary()


#%%


for layer in model.layers[:-12]:
    layer.trainable = False 
    #model.layers.pop()
model.summary()


#%%
# predict whole task model

input_test_task = Input(shape=(30, 30, 1), name='test_input')
input_frozen_model = concatenate(model.output, name='concat_frozen_layers')

# convolution layers
x_test = Conv2D(64, (3, 3), activation='relu', name='test_convolution')(input_test_task)
x_test = MaxPooling2D(pool_size=(2, 2), name='test_pooling')(x_test)
x_test = Dropout(0.25, name='test_dropout')(x_test)
x_test = Flatten(name='test_flatten')(x_test)

# merge frozen layers
merge_frozen = concatenate([
    x_test,
    input_frozen_model
     ], name='concat_test_frozen') 

# out layers
out_final = Dense(128, activation='relu', name='test_out_dense_1')(merge_frozen)
out_final = Dense(128, activation='relu', name='test_out_dense_2')(out_final)

out_final_list = []
for i in range(900):
        out_final_list.append(Dense(10, activation='softmax', name=str('ouput_' + str(i)))(out_final))

new_model = Model(
    inputs=[
        input_test_task,
        model.input], 
    outputs=[
        out_final_list
    ])

opt = Adam(lr=1e-3, decay=1e-3)

new_model.compile(loss="binary_crossentropy", optimizer=opt)

new_model.summary()

#%%
# final task generation

train_input = []
train_output = []
test_input = []
test_output = []

cnt = 0

# use all the tasks in train
# use input and output

for task in train_data:
    for sample in task['train']:
        _input = sample['input']
        _output = sample['output']
                
        if (len(_input) > 1) & (len(_output) > 1):
            train_input.append(enhance_mat_30x30(_input))
            train_output.append(enhance_mat_30x30(_output))
            test_input.append(enhance_mat_30x30(task['test'][0]['input']))
            test_output.append(enhance_mat_30x30(task['test'][0]['output']))
        else:
            cnt += 1

print('Thrown away samples: ')
print(cnt)

print('Total train input samples: ')
print(len(train_input))

print('Total train input samples: ')
print(len(train_output))

print('Total test samples: ')
print(len(test_input))


#%%
# create more training data
import copy 

task_lists = [train_input, train_output, test_input, test_output]
# shift right
def shift_right(task_lists, i=0):

    print('original number of tasks: ', len(task_lists[0]))
    for i_task in range(i, len(task_lists[0])):
        check = []
        for task_list in task_lists:
            # check if column on the right is completely black
            check.append(all([row[-1] == 0 for row in task_list[i_task]]))
        if all(check):
            for task_list in task_lists:
                new_task = copy.deepcopy(task_list[i_task].tolist())
                for row in new_task:
                    row.pop(-1)
                    row.insert(0, [0.0]) # insert zero on first position
                task_list.append(np.array(new_task))
        
    print('new number of tasks: ', len(task_lists[0]))
    return task_lists, i_task

i = 0
for counter in range(30):
    print('\nloop number ', counter)
    task_lists, i = shift_right(task_lists, i)
print('end of task creation')
train_input, train_output, test_input, test_output = task_lists[0], task_lists[1], task_lists[2], task_lists[3]


#%%

plt.plot_matrix(test_input[1].reshape(30, 30))
plt.plot_matrix(train_input[1].reshape(30, 30))
plt.plot_matrix(train_output[1].reshape(30, 30))
plt.plot_matrix(test_output[1].reshape(30, 30))

#%%

plt.plot_matrix(test_input[4677].reshape(30, 30))
plt.plot_matrix(train_input[4677].reshape(30, 30))
plt.plot_matrix(train_output[4677].reshape(30, 30))
plt.plot_matrix(test_output[4677].reshape(30, 30))


#%%

plt.plot_matrix(test_input[7].reshape(30, 30))
plt.plot_matrix(train_input[7].reshape(30, 30))
plt.plot_matrix(train_output[7].reshape(30, 30))
plt.plot_matrix(test_output[7].reshape(30, 30))

#%%


test_output_processed = []
for i in range(900):
    test_output_processed.append([])

for task in test_output:
    all_pixel = to_categorical(task.flatten(), num_classes=10)
    for i in range(900):
        test_output_processed[i].append(all_pixel[i])


#%%
# train final model

history = new_model.fit([
    np.array(test_input),
    np.array(train_input),
    np.array(train_output)
    ],
    test_output_processed,
    epochs=100,
    verbose=0
    )


# %%

history = new_model.history
log = Logger('final_model_unknown_epochs')
log.save_experiment(new_model, history)


#%%

plt.plot_loss(log, 'loss', save=True)


#%%

y_hat = new_model.predict([
    np.array(test_input),
    np.array(train_input),
    np.array(train_output)
    ])

#%%
# post processing 

y_hat_processed = []

for i in range(len(y_hat[0])):
    y_hat_processed.append([])
    for pixel in y_hat:
        y_hat_processed[i].append(np.argmax(pixel[i]))

#%%

plt.plot_matrix(np.array(y_hat_processed[100]).reshape(30, 30))

