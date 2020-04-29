from utils.plotting import plot_matrix
import random
from collections import Counter
from itertools import chain
import copy

# This file will be used to apply various transformations on tasks
# You should use the methods ending with `tasks`
# All the methods will take a task_list, apply transformations
# and return the transformed_task_list + [y_labels]

# currently there to use:
"""
def rotate_tasks(task_list):
    return rotated_task_list, y_rotation
def remove_line_tasks(task_list):
    return removed_line_task_list, y_row_or_column, y_line_nr
def change_least_common_color_tasks(task_list):
    return changed_color_task_list, y_old_color, y_new_color
def change_random_color_tasks(task_list):
    return changed_color_task_list, y_old_color, y_new_color
def shift_line_tasks(task_list):
    return shifted_line_task_list, y_row_or_column, y_line_nr
def multiply_tasks(task_list):
    return multiplied_task_list, y_multiply
def multiply_rotation_tasks(task_list):
    return multiplied_rot_tasks, y_multiplied_rot
def mirror_tasks(task_list):
    return mirrored_task_list, y_mirror
def double_line_with_multiple_colors_tasks(task_list):
    return double_line_task_list, y_row_or_column, y_line_nr
"""

ANGLE_LIST = [
    1, # rotate 90 degrees
    2, # rotate 180 degrees
    3, # rotate 270 degrees
]

ROW_OR_COLUMN = [
    1, # remove row
    2, # remove column
]

# TODO: think if 10 should be here aswell
# TODO: think if 9 should be here
COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

FIFTY_FIFTY = [
    True,
    False,
]

SEVENTYFIVE_PERCENT = [
    True,
    True,
    True,
    False,
]

def get_task_list_copy(task_list):
    return copy.deepcopy(task_list)


# Takes task_list and rotates each task randomly
# Returns list of rotated tasks + list of lables about rotation
def rotate_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    
    rotated_task_list = []
    y_rotation = []
    
    for task in task_list:
        angle = random.choice(ANGLE_LIST)
        y_rotation.append(angle)
        rotated_task_list.append(rotate_task(task=task, angle=angle))
        
    return rotated_task_list, y_rotation

# rotates one task according to spec (clockwise all the time)
# returns rotated task
def rotate_task(task, angle):
    for i in range(angle):
        task = list(zip(*reversed(task)))
    # convert list of tuples to list of lists
    return [list(elem) for elem in task]

# Takes task_list and removes a row or column from every task
# Returns list with altered tasks +
# labels with row or column remvoed +
# index of removed row of column
def remove_line_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    removed_line_task_list = []
    y_row_or_column = []
    y_line_nr = []
    
    for task in task_list:
        row_or_col = random.choice(ROW_OR_COLUMN)
        y_row_or_column.append(row_or_col)
        
        line_nr = 0
        
        if row_or_col == 1:
            line_nr = random.choice(range(len(task)))
        else:
            line_nr = random.choice(range(len(task[0])))
            
        y_line_nr.append(line_nr + 1)
        
        removed_line_task_list.append(remove_line(task, row_or_col, line_nr))
    return removed_line_task_list, y_row_or_column, y_line_nr

# removes a line from a task
# row = true -> row will be removed
# row = false -> column will be removed
# returns task with removed line 
def remove_line(task, row_or_col, line_nr):
    if row_or_col == 1:
        return remove_row(task, line_nr)
    else:
        return remove_column(task, line_nr)

def remove_column(task, column_nr):
    for row in task:
        del row[column_nr]
    return task

def remove_row(task, row_nr):
    task.pop(row_nr)
    return task

# Takes a task list and changes the leas common color of this task
# Returns changed task + y_labels of old colors + y_labels of new colors
def change_least_common_color_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    
    changed_color_task_list = []
    y_old_color = []
    y_new_color = []
    
    for task in task_list:
        old_color = get_least_common_color(task)
        y_old_color.append(old_color)
        
        # to prevent choosing the same color
        new_color = old_color
        while new_color == old_color:
            new_color = random.choice(COLORS)
            
        y_new_color.append(new_color)
        
        changed_task = []
        
        for row in task:
            changed_task.append([el if el != old_color else new_color for el in row])
            
        changed_color_task_list.append(changed_task)
    
    return changed_color_task_list, y_old_color, y_new_color
        
    
# Takes a task and returns it's least common color
# if multiple colors are the leas common ones if will randomly choose one
def get_least_common_color(task):
    counter_obj = Counter(chain.from_iterable(task))
    most_common = counter_obj.most_common()
    
    least_common = min([tuple_el[1] for tuple_el in most_common])
    colors = [tuple_el[0] for tuple_el in most_common if tuple_el[1] == least_common]
    
    return random.choice(colors)


# Takes a task list and changes a random common color of this task
# Returns changed task + y_labels of old colors + y_labels of new colors
def change_random_color_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    
    changed_color_task_list = []
    y_old_color = []
    y_new_color = []
    
    for task in task_list:
        available_colors = set(color for row in task for color in row)
        old_color = random.sample(available_colors, 1)[0]
        y_old_color.append(old_color)
        
        # to prevent choosing the same color
        new_color = old_color
        while new_color == old_color:
            new_color = random.choice(COLORS)
            
        y_new_color.append(new_color)
        
        changed_task = []
        for row in task:
            changed_task.append([el if el != old_color else new_color for el in row])
        changed_color_task_list.append(changed_task)
    
    return changed_color_task_list, y_old_color, y_new_color

# Takes task_list and shifts a row or column from every task
# Returns list with altered tasks +
# labels with row or column shifted +
# index of shifted row of column
def shift_line_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    
    shifted_line_task_list = []
    y_row_or_column = []
    y_line_nr = []
    
    for task in task_list:
        row_or_col = random.choice(ROW_OR_COLUMN)
        y_row_or_column.append(row_or_col)
        
        line_nr = 0
        
        if row_or_col == 1:
            line_nr = random.choice(range(len(task)))
        else:
            line_nr = random.choice(range(len(task[0])))
            
        y_line_nr.append(line_nr + 1)
        shifted_line_task_list.append(shift_line(task, row_or_col, line_nr))
    return shifted_line_task_list, y_row_or_column, y_line_nr


# shifts a line from a task
# row = true -> row will be shifted
# row = false -> column will be shifted
# returns task with shifted line 
def shift_line(task, row_or_col, line_nr):
    if row_or_col == 1:
        return shift_row(task, line_nr)
    else:
        return shift_column(task, line_nr)

def shift_row(task, row_nr):
    task[row_nr].insert(0, task[row_nr].pop())
    return task

def shift_column(task, col_nr):
    task_rot = rotate_task(task, angle=1)
    task_rot[col_nr].insert(0, task_rot[col_nr].pop())
    return rotate_task(task_rot, angle=3)


# Takes a task list
# Multiplies this task 2 or 4 times (if size is possible < 30)
# Result ->
# | task | task |

# or
# | task |
# | task |

#or
# | task | task |
# | task | task |
# returns altered task list + 
# y_labels with 1, 2, 4 (how often task is in altered task)
def multiply_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    multiplied_task_list = []
    
    y_multiply = []
    
    for task in task_list:
        factor = 0
        if len(task) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = multiply_row(task)
                factor = factor + 1
        if len(task[0]) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = multiply_col(task)
                factor = factor + 1

        multiplied_task_list.append(task)
        y_multiply.append(factor)
    return multiplied_task_list, y_multiply
    
def multiply_row(task):
    return task + task

def multiply_col(task):
    task = rotate_task(task, angle=1)
    task = multiply_row(task)
    return rotate_task(task, angle=3)

# Works like multiply_tasks
# But it rotatets the tasks to mirror them
def multiply_rotation_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    multiplied_rot_tasks = []
    
    y_multiplied_rot = []
    
    for task in task_list:
        factor = 0
        if len(task) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = multiply_rotation_row(task)
                factor = factor + 1
        if len(task[0]) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = multiply_rotation_col(task)
                factor = factor + 1

        multiplied_rot_tasks.append(task)
        y_multiplied_rot.append(factor)
    return multiplied_rot_tasks, y_multiplied_rot
    
def multiply_rotation_row(task):
    return task + rotate_task(task, angle=2)

def multiply_rotation_col(task):
    task = rotate_task(task, angle=1)
    task = multiply_rotation_row(task)
    return rotate_task(task, angle=3)

# works like multiply
# but mirrors the task
# which will result in symmetric output pictures
def mirror_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    mirrored_task_list = []
    
    y_mirror = []
    
    for task in task_list:
        factor = 0
        if len(task) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = mirror_row(task)
                factor = factor + 1
        if len(task[0]) < 16:
            if random.choice(SEVENTYFIVE_PERCENT):
                task = mirror_col(task)
                factor = factor + 1

        mirrored_task_list.append(task)
        y_mirror.append(factor)
    return mirrored_task_list, y_mirror

def mirror_row(task):
    return task + task[::-1]
    
def mirror_col(task):
    task = rotate_task(task, angle=1)
    task = mirror_row(task)
    return rotate_task(task, angle=3)

def double_line_with_multiple_colors_tasks(task_list):
    task_list = get_task_list_copy(task_list)
    double_line_task_list = []
    y_row_or_column = []
    y_line_nr = []
    
    for task in task_list:
        row_or_col = random.choice(ROW_OR_COLUMN)
        
        
        possible_lines = []
        
        if row_or_col == 1:
            possible_lines = get_row_with_multiple_colors(task)
            if len(possible_lines) == 0:
                possible_lines = get_col_with_multiple_colors(task)
                row_or_col = 2
        else:
            possible_lines = get_col_with_multiple_colors(task)
            if len(possible_lines) == 0:
                possible_lines = get_row_with_multiple_colors(task)
                row_or_col = 1
        
        y_row_or_column.append(row_or_col)
            
        line_nr = random.choice(possible_lines)
        y_line_nr.append(line_nr)
        
        double_line_task_list.append(double_line(row_or_col, task, line_nr))
    return double_line_task_list, y_row_or_column, y_line_nr
    
def get_row_with_multiple_colors(task):
    line_nr = []
    for idx, row in enumerate(task):
        if len(set(row)) > 1:
            line_nr.append(idx)
    return line_nr

def get_col_with_multiple_colors(task):
    return get_row_with_multiple_colors(rotate_task(task, angle=1))

def double_line(row_or_col, task, line_nr):
    if row_or_col == 1:
        return double_row(task, line_nr)
    else:
        return double_col(task, line_nr)

def double_row(task, row_nr):
    task.insert(row_nr, task[row_nr])
    return task

def double_col(task, col_nr):
    rot_task = double_row(rotate_task(task, angle=1), col_nr)
    return rotate_task(rot_task, angle=3)
    
if __name__ == "__main__":
    test_task_1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    test_task_2 = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    test_task_3 = [[1, 2, 3], [2, 2, 3], [3, 3, 3]]
    
    test_task_list = [test_task_1, test_task_2, test_task_3]
    print(test_task_list)

    print("Rotation:")
    rotated_test_task_list, y_rotation = rotate_tasks(test_task_list)
    #plot_matrix(test_task_list[0])
    #plot_matrix(rotated_test_task_list[0])    
    print(rotated_test_task_list)
    print(y_rotation)
    
    print("Remove Row or Col:")
    removed_line_test_task_list, y_row_or_column, y_line_nr = remove_line_tasks(test_task_list)
    #plot_matrix(test_task_list[0])
    #plot_matrix(removed_line_test_task_list[0])
    print(removed_line_test_task_list)
    print(y_row_or_column)
    print(y_line_nr)
    
    print("Change least common color:")
    changed_least_color_task_list, y_old_color, y_new_color = change_least_common_color_tasks(test_task_list)
    print(changed_least_color_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(changed_least_color_task_list[2])
    print(y_old_color)
    print(y_new_color)
    
    print("Change random color in task:")
    change_random_color_task_list, y_old_color, y_new_color = change_random_color_tasks(test_task_list)
    print(change_random_color_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(change_random_color_task_list[2])
    print(y_old_color)
    print(y_new_color)
    
    print("Shift Row or Col:")    
    shifted_line_test_task_list, y_row_or_column, y_line_nr = shift_line_tasks(test_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(shifted_line_test_task_list[2])
    print(shifted_line_test_task_list)
    print(y_row_or_column)
    print(y_line_nr)
    
    print("Multiply task:")
    multiplied_test_task_list, y_multiply = multiply_tasks(test_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(multiplied_test_task_list[2])
    print(multiplied_test_task_list)
    print(y_multiply)
    
    print("Multiply with rotation task:")
    multiplied_rot_test_task_list, y_multiplied_rot = multiply_rotation_tasks(test_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(multiplied_rot_test_task_list[2])
    print(multiplied_rot_test_task_list)
    print(y_multiplied_rot)
    
    print("Multiply with rotation task:")
    mirrored_test_task_list, y_mirror = mirror_tasks(test_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(mirrored_test_task_list[2])
    print(mirrored_test_task_list)
    print(y_mirror)    
    
    print("Double line with multiple colors:")
    doubled_line_test_task_list, y_row_or_column, y_line_nr = double_line_with_multiple_colors_tasks(test_task_list)
    #plot_matrix(test_task_list[2])
    #plot_matrix(doubled_line_test_task_list[2])
    print(doubled_line_test_task_list)
    print(y_row_or_column)
    print(y_line_nr)

    # To verify that the original list is unchanged
    print('Should be unchanged: ')
    print(test_task_list)
    
    assert test_task_list == [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[3, 3, 3], [2, 2, 2], [1, 1, 1]], [[1, 2, 3], [2, 2, 3], [3, 3, 3]]]