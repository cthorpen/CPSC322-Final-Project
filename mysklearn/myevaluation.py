from mysklearn import myutils
import numpy as np
import math

# TODO: copy your myevaluation.py solution from PA5-6 here


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    print('maybe:', random_state)

    # random number seed with random_state
    if random_state is not None:
        np.random.seed(random_state)
    # randomize order of lists if shuffle is True
    if shuffle is True:
        X, y = myutils.randomize_in_place(X, y)
    # convert float test_size to int, to make it iterable
    if type(test_size) is float and test_size < 1.0:
        test_size = math.ceil(test_size * len(X))
    # separate list into train/test
    for i in range(len(X)):
        # if index lower than test_size -> train
        if i < len(X) - test_size:
            X_train.append(X[i])
            y_train.append(y[i])
        # if higher -> test
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    return X_train, X_test, y_train, y_test


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    all_folds = []

    # random number seed with random_state
    if random_state is not None:
        np.random.seed(random_state)

    # create all folds
    for _ in range(n_splits):
        all_folds.append([])

    # create list of X indices
    X_idxs = []
    for i in range(len(X)):
        X_idxs.append(i)

    # randomize order of lists if shuffle is True
    if shuffle is True:
        X_idxs, _ = myutils.randomize_in_place(X_idxs)

    # put indices in folds
    # (this ends up being test folds)
    for i in range(len(X_idxs)):
        fold_idx = i % len(all_folds)
        all_folds[fold_idx].append(X_idxs[i])
    X_test_folds = all_folds

    for i in range(len(X_test_folds)):
        training_fold = []
        for j in range(len(X)):
            if j not in X_test_folds[i]:
                training_fold.append(j)
        X_train_folds.append(training_fold)

    return X_train_folds, X_test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_train_folds = []
    X_test_folds = []

    # random number seed with random_state
    if random_state is not None:
        np.random.seed(random_state)

    # create list of X indices
    X_idxs = []
    for i in range(len(X)):
        X_idxs.append([i])
    # randomize order of lists if shuffle is True
    if shuffle is True:
        X_idxs, _ = myutils.randomize_in_place(X_idxs)

    # append class labels to end of X
    if y is not None:
        for i in range(len(X_idxs)):
            X_idxs[i].append(y[i])

    # create header for grouping
    header = []
    for _ in range(len(X_idxs[0])):
        header.append("feature")
    header[-1] = "label"
    # group by class label
    _, group_subtables = myutils.group_by(X_idxs, header, "label")

    # create all folds
    all_folds = []
    for _ in range(n_splits):
        all_folds.append([])

    for table in group_subtables:
        for i in range(len(table)):
            fold_idx = i % len(all_folds)
            all_folds[fold_idx].append(table[i][0])
    X_test_folds = all_folds

    for i in range(len(X_test_folds)):
        training_fold = []
        for table in group_subtables:
            for j in range(len(table)):
                if table[j][0] not in X_test_folds[i]:
                    training_fold.append(table[j][0])
        X_train_folds.append(training_fold)

    return X_train_folds, X_test_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    # random number seed with random_state
    if random_state is not None:
        np.random.seed(random_state)

    # check if y is none
    if y is None:
        y_sample = None
        y_out_of_bag = None

    # get number of samples to generate
    if n_samples is None:
        n_samples = len(X)
    # print("n_samples", n_samples)

    # generate samples
    for _ in range(n_samples):
        rand_idx = np.random.randint(len(X))
        # print("rand_idx", rand_idx)
        X_sample.append(X[rand_idx])
        if y is not None:
            y_sample.append(y[rand_idx])

    # out of bag samples
    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
        if y is not None:
            if y.index(y[i]) not in y_sample:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # create empty matrix
    for _ in labels:
        matrix.append([])
    # fill matrix with empty values
    for i in range(len(matrix)):
        for _ in labels:
            matrix[i].append(0)
    # increment item in matrix based on pred vs actual
    for i in range(len(y_true)):
        # print("len y true[i]", len(y_true[i]))
        # print("len y pred[i]", len(y_pred[i]))
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    score = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            score += 1

    if normalize is True:
        return (score / len(y_pred))

    return score


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # check if labels is None
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    # check if pos_label is None
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0
    for i in range(len(y_pred)):
        # check if it is a true positive
        if y_pred[i] == pos_label and y_pred[i] == y_true[i]:
            tp += 1
        # check if it is a false positive
        elif y_pred[i] == pos_label and y_pred[i] != y_true[i]:
            fp += 1

    # check for division by 0
    if (tp + fp == 0):
        return 0.0

    # normal return case
    return (tp / (tp + fp))


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    # check if labels is None
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    # check if pos_label is None
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0
    for i in range(len(y_pred)):
        # check if it is a true positive
        if y_pred[i] == pos_label and y_pred[i] == y_true[i]:
            tp += 1
        # check if it is a false negative
        if y_pred[i] != pos_label and y_pred[i] != y_true[i]:
            fn += 1

    # check for division by 0
    if (tp + fn == 0):
        return 0.0

    # normal return case
    return (tp / (tp + fn))


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    # get precision
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    # get recall
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    # check for division by 0
    if (precision + recall) == 0:
        return 0.0

    return (2 * (precision * recall) / (precision + recall))
