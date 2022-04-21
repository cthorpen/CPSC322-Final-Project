import numpy as np
import csv
from . import myevaluation
from . import myclassifiers
from tabulate import tabulate

# TODO: your reusable general-purpose functions here

# adding random_state here


def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap values with
        rand_index = np.random.randint(0, len(alist))  # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
    return alist, parallel_list


def get_tables(filename):
    """Load table and header from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            table(2D List): list of data
            header(list): column names
        """
    # initialize header and table
    header = []
    table = []
    # open file and copy values into header and table
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            table.append(row)
    csvfile.close()
    return table, header


def group_by(table, header, groupby_col_name):
    groupby_col_index = header.index(groupby_col_name)  # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col)))  # [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for row in table:
        groupby_val = row[groupby_col_index]  # e.g. this row's modelyear
        # which subtable does this row belong to?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(
            row.copy())  # make copy

    return group_names, group_subtables


def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col


def compute_euclidean_distance(v1, v2):
    """ function to compute euclidean distance
    Args: v1, v2 (list of int): list of points to determine distance between
    Return: euclidean distance
    """
    # print("v1:", v1)
    # print("v2:", v2)
    dist = 0
    for i in range(len(v1)):
        # if attribute is numeric
        if isinstance(v1[i], int) and isinstance(v2[i], int) or isinstance(v1[i], float) and isinstance(v2[i], float):
            dist += ((v1[i] - v2[i]) ** 2)
        else:
            if v1[i] != v2[i]:
                dist += 1
            # print(dist)
    return np.sqrt(dist)
    # return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v2))])) # OG way


def get_most_frequent(list):
    """ function to find the most frequent element in a list. turn list into a set to 
        remove any duplicates, then find the highest count of occurences (key= occurence count)
    Args: list (list of any): list to search for the most frequent element in
    Return: most_freq: most frequent element
    """
    most_freq = max(set(list), key=list.count)
    return most_freq


def continuous_to_categorical(con_attr):
    """Creates a categorical attribute from a continuous attribute
    Args:
        con_attr(list): continuous attribute (column) from a dataset
    Returns:
        cat_attr(list): parallel list to con_attr with corresponding categorical ratings
    Notes:
        created for auto dataset and rating vehicles based on their MPG
    """
    cat_attr = []
    for item in con_attr:
        if item >= 45:
            cat_attr.append(10)
        elif item >= 37:
            cat_attr.append(9)
        elif item >= 31:
            cat_attr.append(8)
        elif item >= 27:
            cat_attr.append(7)
        elif item >= 24:
            cat_attr.append(6)
        elif item >= 20:
            cat_attr.append(5)
        elif item >= 17:
            cat_attr.append(4)
        elif item >= 15:
            cat_attr.append(3)
        elif item == 14:
            cat_attr.append(2)
        else:
            cat_attr.append(1)
    return cat_attr


def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
    col.sort()  # inplace sort
    values = []
    counts = []
    for value in col:
        if value in values:
            counts[-1] += 1  # okay because col is sorted
        else:  # haven't seen this value before
            values.append(value)
            counts.append(1)
    return values, counts


def get_sets_from_folds(X, y, folds):
    X_sets = []
    y_sets = []
    for fold in folds:
        X_train = []
        y_train = []
        for i in range(len(fold)):
            X_train.append(X[fold[i]])
            y_train.append(y[fold[i]])
        X_sets.append(X_train)
        y_sets.append(y_train)
    return X_sets, y_sets


def create_empty_matrix(labels):
    matrix = []
    for i in range(len(labels)):
        matrix.append([])
        for j in range(len(labels)):
            matrix[i].append(0)
    return matrix


def all_same_class(att_partition):
    """ determine of all instances of a partition have same class label

        Args:
            att_partition (list of obj): list of instances to check class labels

        Returns:
            True/False: if all class labels are the same
    """
    # if all index -1 are same, return true
    class_label = att_partition[0][-1]
    for att in att_partition:
        if att[-1] != class_label:
            return False
    return True


def fit_predict_classification(X_train_sets, y_train_sets, X_test_sets, y_test_sets, classifier, name, labels):
    # metric variables
    dm_acc, dm_prec, dm_rcl, dm_f1 = 0, 0, 0, 0
    # create empty matrix
    dm_matrix = create_empty_matrix(labels)

    # fit and predict for all training and testing sets
    for i in range(len(X_train_sets)):
        # fit model
        classifier.fit(X_train_sets[i], y_train_sets[i])
        # predict
        y_pred = classifier.predict(X_test_sets[i])
        # metrics
        dm_acc += myevaluation.accuracy_score(y_train_sets[i], y_pred, True)
        dm_rcl += myevaluation.binary_recall_score(
            y_test_sets[i], y_pred, labels, pos_label=None)  # change these to correct positive label
        dm_prec += myevaluation.binary_precision_score(
            y_test_sets[i], y_pred, labels, pos_label=None)
        dm_f1 += myevaluation.binary_f1_score(
            y_test_sets[i], y_pred, labels, pos_label=None)
        mtx = myevaluation.confusion_matrix(y_test_sets[i], y_pred, labels)
        for i in range(len(dm_matrix)):
            for j in range(len(dm_matrix)):
                dm_matrix[i][j] += mtx[i][j]

    dm_acc = round(dm_acc / len(X_train_sets), 2)
    dm_err = round(1 - dm_acc, 2)
    dm_rcl = round(dm_rcl / len(X_train_sets), 2)
    dm_prec = round(dm_prec / len(X_train_sets), 2)
    dm_f1 = round(dm_f1 / len(X_train_sets), 2)

    print("============================")
    print(name, "Classification")
    print("============================")
    print("Accuracy:", dm_acc, "~", "Error Rate:", dm_err)
    print("Recall Score:", dm_rcl)
    print("Precision Score:", dm_prec)
    print("F1 Score:", dm_f1)
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(tabulate(dm_matrix, headers=labels, tablefmt="pretty"))
    return
