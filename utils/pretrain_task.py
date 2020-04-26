from plotting import plot_matrix
import random
from collections import Counter
from itertools import chain
import copy

# This file will be used to apply various transformations on tasks
# You should use the methods ending with `tasks`
# All the methods will take a task_list, apply transformations
# and return the transformed_task_list + [y_labels]

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
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

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
        
        if row_or_col == 1:
            line_nr = random.choice(range(len(task[0])))
        else:
            line_nr = random.choice(range(len(task)))
            
        y_line_nr.append(line_nr)
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
    
    
    # To verify that the original list is unchanged
    print(test_task_list)
    

    
