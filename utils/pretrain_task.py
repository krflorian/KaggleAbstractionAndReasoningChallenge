from plotting import plot_matrix
import random

ANGLE_LIST = [
    1, # rotate 90 degrees
    2, # rotate 180 degrees
    3, # rotate 270 degrees
]

ROW_OR_COLUMN = [
    1, # remove row
    2, # remove column
]

# Takes task_list and rotates each task randomly
# Returns list of rotated tasks + list of lables about rotation
def rotate_tasks(task_list):
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
    
if __name__ == "__main__":
    test_task_1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    test_task_2 = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    
    test_task_list = [test_task_1, test_task_2]
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
    

    
