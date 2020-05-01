

#tensorflow
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Subtract, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility



def create_model()

    # Raphael Model
    # input
    input_ = Input(shape=(30, 30, 1), name='train_input')
    output_ = Input(shape=(30, 30, 1), name='train_output')

# convolution layers
x_1 = Conv2D(64, (3, 3), activation='relu')(input_)
x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
x_1 = Dropout(0.25)(x_1)
x_1 = Flatten()(x_1)

x_2 = Conv2D(64, (3, 3), activation='relu')(output_)
x_2 = MaxPooling2D(pool_size=(2, 2))(x_2)
x_2 = Dropout(0.25)(x_2)
x_2 = Flatten()(x_2)

merge = concatenate([x_1, x_2])

merge = Dense(128, activation='relu')(merge)
merge = Dense(128, activation='relu')(merge)
merge = Dropout(0.3)(merge)

pretrain.mirror_tasks,
pretrain.double_line_with_multiple_colors_tasks,

# regression layers
out_1 = Dense(128, activation='relu')(merge)
out_1 = Dense(1, activation='linear', name='rows')(out_1)

out_2 = Dense(128, activation='relu')(merge)
out_2 = Dense(1, activation='linear', name='cols')(out_2)

out_9 = Dense(128, activation='relu')(merge)
out_9 = Dense(1, activation='linear', name='removed_line_nr')(out_9)

out_11 = Dense(128, activation='relu')(merge)
out_11 = Dense(1, activation='linear', name='shifted_line_nr')(out_11)

# out_15 = Dense(128, activation='relu')(merge)
# out_15 = Dense(1, activation='linear', name='doubled_line_nr')(out_15)

# multi-label classification layers
out_4 = Dense(128, activation='relu')(merge)
out_4 = Dense(4, activation='sigmoid', name='rotation_angle')(out_4)

out_5 = Dense(128, activation='relu')(merge)
out_5 = Dense(3, activation='sigmoid', name='multiply_factor')(out_5)

out_6 = Dense(128, activation='relu')(merge)
out_6 = Dense(10, activation='sigmoid', name='changed_color_old')(out_6)

out_7 = Dense(128, activation='relu')(merge)
out_7 = Dense(10, activation='sigmoid', name='changed_color_new')(out_7)

out_8 = Dense(128, activation='relu')(merge)
out_8 = Dense(3, activation='sigmoid', name='removed_row_or_column')(out_8)

out_10 = Dense(128, activation='relu')(merge)
out_10 = Dense(3, activation='sigmoid', name='shifted_row_or_column')(out_10)

out_12 = Dense(128, activation='relu')(merge)
out_12 = Dense(3, activation='sigmoid', name='multiply_rotation_factor')(out_12)

out_13 = Dense(128, activation='relu')(merge)
out_13 = Dense(3, activation='sigmoid', name='multiply_mirror_factor')(out_13)

# out_14 = Dense(128, activation='relu')(merge)
# out_14 = Dense(3, activation='sigmoid', name='doubled_row_or_column')(out_14)


model = Model(inputs=[input_, output_], outputs=[
    out_1, out_2, out_4,
    out_5, out_6, out_7,
    out_8, out_9, out_10,
    out_11, out_12, out_13,
    # out_14, out_15
    ])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "rows": "mean_absolute_error",
    "cols": "mean_absolute_error",
    "removed_line_nr": "mean_absolute_error",
    "shifted_line_nr": "mean_absolute_error",
    "rotation_angle": "binary_crossentropy",
    "multiply_factor": "binary_crossentropy",
    "changed_color_old": "binary_crossentropy",
    "changed_color_new": "binary_crossentropy",
    "removed_row_or_column": "binary_crossentropy",
    "shifted_row_or_column": "binary_crossentropy",
    "multiply_rotation_factor": "binary_crossentropy",
    "multiply_mirror_factor": "binary_crossentropy",
    # "doubled_row_or_column": "binary_crossentropy",
    # "doubled_line_nr": "mean_absolute_error"
    }

model.compile(loss=losses, optimizer=opt)


