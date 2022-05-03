import collections
import itertools
import random

import numpy as np

from mysklearn import myevaluation
from . import myutils
from math import log2
import operator
import copy

# TODO: copy your myclassifiers.py solution from PA4-6 here (BELOW)


class MyRandomForestClassifier:

    def __init__(self, N=3, M=2, F=1):
        self.N = N
        self.M = M
        self.F = F
        self.X_train = None
        self.y_train = None
        self.trees = None
        # extra
        self.header = None
        self.domains = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        self.X_train = X_train
        self.y_train = y_train
        # make generic header of attributes and dictionary from header (attribute domains) for entropy
        self.header, self.domains = self.create_header_and_domains()

        X_remainder, X_test, y_remainder, y_test = myevaluation.train_test_split(
            X_train, y_train)

        # stitching X_train and y_train together
        train = [X_remainder[i] + [y_remainder[i]]
                 for i in range(len(X_remainder))]

        # make copy of header list since tdidt() will modify the list
        available_attributes = self.header.copy()

        # create N trees
        all_trees = []
        for _ in range(self.N):
            new_tree = self.tdidt(train, available_attributes)
            all_trees.append(new_tree)

        # print(len(all_trees))

        accuracies = []
        for tree in all_trees:
            total_count = len(y_test)
            predictions = self.single_predict(tree, X_test)
            correct_count = 0
            for i, pred in enumerate(predictions):
                if pred == y_test[i]:
                    correct_count += 1
            accuracy = float(correct_count) / float(total_count)
            accuracies.append(accuracy)

        # print(accuracies)

        accuracies = sorted(accuracies)
        accuracies = accuracies[(len(accuracies) - self.M):]

        # print(accuracies)

        chosen_trees = []
        for accuracy in accuracies:
            index = accuracies.index(accuracy)
            chosen_trees.append(all_trees[index])
        self.trees = chosen_trees

        # for tree in self.trees:
        #     print(tree)

        pass  # TODO: fix this

    def single_predict(self, tree, X_test):
        """Makes SINGLE prediction for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            predictions.append(self.tdidt_predict(tree, instance))
        return predictions  # TODO: fix this

    def predict(self, X_test):
        """Makes ALL predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # gather all predictions from the testing set
        y_pred = []
        for tree in self.trees:
            pred = []
            for instance in X_test:
                pred.append(self.tdidt_predict(tree=tree, instance=instance))
            y_pred.append(pred)

        # print(y_pred)

        # avg_y_pred = []
        # for y in y_pred:
        #     avg_y_pred.append(myutils.get_most_frequent(y))

        # print(avg_y_pred)

        avg_preds = []
        for instance in X_test:
            avg_preds.append([])
        for j, prediction in enumerate(y_pred):
            for i, inst in enumerate(prediction):
                avg_preds[i].append(inst)
        mean_preds = []
        for pred in avg_preds:
            # print("pred---", pred)
            mean_preds.append(myutils.get_most_frequent(pred))
        return mean_preds

    # BELOW: helper methods for above

    def tdidt_predict(self, tree, instance):
        """ Recursively traverses the tree to find the best fitted prediction for given X_test instances

        ARGS:
            tree (list/tree of obj): current subtree being traversed
            header (list of str): list of (generic) attribute names
            instance (list of obj): current instance of X_test being predicted on

        RETURNS:
            tree[i] (list of obj): current location in subtree being traversed (attribute node or leaf node)
        """
        info_type = tree[0]
        # we are at leaf node (base case)
        if info_type == "Leaf":
            return tree[1]
        # we need to match the attribute's value in the
        # instance with appropriate value list in the tree
        # for loop that traverses thru each value list, recurse on match with instance's value
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # edge matching
            if value_list[1] == instance[att_index]:
                # match! recurse
                return self.tdidt_predict(value_list[2], instance)

    def bootstrap_sample(self, data):
        ''' Function for bootstrap method
            Self is passed in table 
        '''
        sample = []
        used_indexs = []
        for _ in range(self.F):
            rand_index = random.randint(0, len(data) - 1)
            if rand_index not in used_indexs:
                sample.append(data[rand_index])
                used_indexs.append(rand_index)

        return sample

    def tdidt(self, current_instances, available_attributes):
        """ recursive function to create a tree from the training data

        Args:
            current_instances (list of list of values): available training data to append to the tree
            available_instances (list of str): list of attributes available to split on
            header (list of str): names of attributes
            attribute_domains (dict of list): dictionary of attribute indices with possible attribute values

        Returns:
            tree (tree as list): current tree created by tdidt
        """

        available_attributes = self.bootstrap_sample(available_attributes)

        # select attribute to split on
        attribute = self.select_attribute(
            current_instances, available_attributes)
        # can't split on this attribute again
        available_attributes.remove(attribute)
        # start to build the tree
        tree = ["Attribute", attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]

            # CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and myutils.all_same_class(att_partition):
                # leaf node, all same class
                leaf = ["Leaf", att_partition[0][-1],
                        len(att_partition), len(current_instances)]
                value_subtree.append(leaf)

            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # mex of class labels, handle clash with majority vote leaf node (from the partition)
                vote = self.majority_vote_leaf_node(att_partition)
                leaf = ["Leaf", vote, len(
                    att_partition), len(current_instances)]
                # print("case 2, leaf:", leaf)
                value_subtree.append(leaf)

             # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # TODO: "backtrack" and replace this attribute node with a majority vote leaf node
                # THE VOTE SWITCHES BETWEEN YES AND NO FOR IPHONE DATA
                vote = self.majority_vote_leaf_node(current_instances)
                leaf = ["Leaf", vote, len(
                    current_instances), len(current_instances)]
                # print("case 3")
                return leaf

            # recurse if none of the previous conditions were met
            else:
                subtree = self.tdidt(
                    att_partition, available_attributes.copy())
                value_subtree.append(subtree)

            tree.append(value_subtree)
            # print("tree:", tree)

        return tree

    def select_attribute(self, instances, attributes):
        """ recursive function to create a tree from the training data

        Args:
            instances (list of obj): instances used to find entropy of thier attribute values
            attributes (list of str): available attributes to split on
            header (list of str): attribute names

        Returns:
           att_to_split (str): attribute with ideal entropy to split on
        """
        Enew = 1.1  # so the calculated Enew will always be smaller
        att_to_split = ""
        # for each available attribute
        for attr in attributes:
            entropy_list, num_inst_list = [], []
            cur_Enew = 0
            partition = self.partition_instances(
                instances, attr)
            # for each value in the attributes domain
            for val in partition:
                # compute entropy of all instances in val
                entropy = self.compute_entropy(
                    partition[val], len(partition[val]))
                # add to lists for finding Enew
                entropy_list.append(entropy)
                num_inst_list.append(len(partition[val]) / len(instances))
            # compute Enew, the weighted sum of all partition entropies
            cur_Enew = self.compute_Enew(entropy_list, num_inst_list)
            # split on attribute with the smallest Enew
            if cur_Enew < Enew:
                Enew = cur_Enew
                att_to_split = attr
        return att_to_split

    def compute_entropy(self, instances, num_instances):
        """ computes entropy of an attribute

        Args:
            instances (list of obj): instances to compute entropy on
            num_instances (int): number of instances with same attribute value

        Returns:
            entropy (double): entropy for an attribute in the instances
        """
        # least frequent class label, considered the 'negative' label
        negative_class_label = collections.Counter(
            self.y_train).most_common()[-1][0]
        num_falses = 0
        for instance in instances:
            # WILL PROBABLY NEED TO BE CHANGED LATER
            if instance[-1] == negative_class_label:
                num_falses += 1
        num_trues = num_instances - num_falses
        if num_falses == 0 or num_trues == 0:
            return 0
        return ((-num_falses/num_instances) * log2(num_falses/num_instances)) - ((num_trues/num_instances) * log2(num_trues/num_instances))

    def compute_Enew(self, entropy_list, num_inst_list):
        """ determines the most frequent class label in a list of instances

        Args:
            entropy_list (list of int): entropy values for an attribute
            num_inst_list (list of double): proportion of attributes with same value (paarllel to entropy_list)

        Returns:
            label (str): most frequent class label
        """
        Enew = 0
        for i in range(len(entropy_list)):
            Enew += entropy_list[i] * num_inst_list[i]
        return Enew

    def partition_instances(self, instances, split_attribute):
        """ group instances by attribute domain values

        Args:
            instances (list of obj): available instances to be partitioned
            split_attribute (str): attribute to split on
            header (list of str): attribute names

        Returns:
            partitions (dict of list): instances partitioned by attribute values
        """
        partitions = {}  # key (attr value): value (subtable)
        att_index = self.header.index(split_attribute)  # e.g. level -> 0
        att_domain = self.domains[att_index]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def majority_vote_leaf_node(self, instances):
        """ determines the most frequent class label in a list of instances

        Args:
            instances (list of obj): instances to check most frequent class labels

        Returns:
            label (obj): most frequent class label
        """
        class_labels = [instances[i][-1] for i in range(len(instances))]
        class_labels = sorted(class_labels)
        label = max(class_labels, key=class_labels.count)
        return label

    def create_header_and_domains(self):
        """Creates a header list and domain dictionary from self.X_train

        Returns:
            header (list of str): list of attributes
            domains (dict of list): dictionary of attribute indices with possible attribute values
        """
        header = []
        domains = {}
        # create header list and initialize domains
        for i in range(len(self.X_train[0])):
            header.append("att" + str(i))
            domains[i] = []
        # fill domains
        for X in self.X_train:
            for i in range(len(X)):
                # print(domains[i])
                if X[i] not in domains[i]:
                    domains[i].append(X[i])
                domains[i].sort()
        return header, domains


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        # additional class attributes for ease
        self.header = None
        self.domains = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # BE MINDFUL OF THE ORDER OF THE TREE

        self.X_train = X_train
        self.y_train = y_train
        self.tree = []
        # make generic header of attributes and dictionary from header (attribute domains) for entropy
        self.header, self.domains = self.create_header_and_domains()
        # stitching X_train and y_train together
        train = [self.X_train[i] + [self.y_train[i]]
                 for i in range(len(self.X_train))]
        # make copy of header list since tdidt() will modify the list
        available_attributes = self.header.copy()
        # create tree using tdidt
        self.tree = self.tdidt(
            train, available_attributes)

        pass  # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        # append each prediction to y_predicted
        for instance in X_test:
            y_predicted.append(self.tdidt_predict(self.tree, instance))
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # no rules of no tree
        if self.tree == None:
            return

        # set header if attribute names given
        new_header = self.header
        if attribute_names != None:
            if len(attribute_names) == len(self.header):
                new_header = attribute_names

        # get domains and domain names
        domain_list = []
        domain_name_list = []
        for att_val, domain in self.domains.items():
            domain_list.append(domain)
            domain_name_list.append("att" + str(att_val))

        # get set of all rules (couldnt figure this out w/out itertools :( )
        rules = list(itertools.product(*domain_list))

        # fill each rule for all combinations of attributes
        for rule in rules:
            rules_sorted = []
            # sort rules in order
            for i in range(len(self.header)):
                att_index = domain_name_list.index(self.header[i])
                rules_sorted.append(rule[att_index])
            print("IF ", end="")
            # get attributes and values
            for i in range(len(self.header)):
                if i != len(self.header) - 1:
                    print(str(new_header[i]) + " == " +
                          str(rules_sorted[i]) + " AND ", end="")
            # class label and value
            print(str(new_header[len(self.header) - 1]) + " == " +
                  str(rules_sorted[len(self.header) - 1]) + " THEN " + class_name + " == ", end="")
            print(self.predict([rules_sorted])[0])

    # BONUS method

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass  # TODO: (BONUS) fix this

    # BELOW: helper methods for above

    def tdidt_predict(self, tree, instance):
        """ Recursively traverses the tree to find the best fitted prediction for given X_test instances

        ARGS:
            tree (list/tree of obj): current subtree being traversed
            header (list of str): list of (generic) attribute names
            instance (list of obj): current instance of X_test being predicted on

        RETURNS:
            tree[i] (list of obj): current location in subtree being traversed (attribute node or leaf node)
        """
        info_type = tree[0]
        # we are at leaf node (base case)
        if info_type == "Leaf":
            return tree[1]
        # we need to match the attribute's value in the
        # instance with appropriate value list in the tree
        # for loop that traverses thru each value list, recurse on match with instance's value
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # edge matching
            if value_list[1] == instance[att_index]:
                # match! recurse
                return self.tdidt_predict(value_list[2], instance)

    def tdidt(self, current_instances, available_attributes):
        """ recursive function to create a tree from the training data

        Args:
            current_instances (list of list of values): available training data to append to the tree
            available_instances (list of str): list of attributes available to split on
            header (list of str): names of attributes
            attribute_domains (dict of list): dictionary of attribute indices with possible attribute values

        Returns:
            tree (tree as list): current tree created by tdidt
        """

        # select attribute to split on
        attribute = self.select_attribute(
            current_instances, available_attributes)
        # can't split on this attribute again
        available_attributes.remove(attribute)
        # start to build the tree
        tree = ["Attribute", attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]

            # CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and myutils.all_same_class(att_partition):
                # leaf node, all same class
                leaf = ["Leaf", att_partition[0][-1],
                        len(att_partition), len(current_instances)]
                value_subtree.append(leaf)

            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # mex of class labels, handle clash with majority vote leaf node (from the partition)
                vote = self.majority_vote_leaf_node(att_partition)
                leaf = ["Leaf", vote, len(
                    att_partition), len(current_instances)]
                # print("case 2, leaf:", leaf)
                value_subtree.append(leaf)

             # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # TODO: "backtrack" and replace this attribute node with a majority vote leaf node
                # THE VOTE SWITCHES BETWEEN YES AND NO FOR IPHONE DATA
                vote = self.majority_vote_leaf_node(current_instances)
                leaf = ["Leaf", vote, len(
                    current_instances), len(current_instances)]
                # print("case 3")
                return leaf

            # recurse if none of the previous conditions were met
            else:
                subtree = self.tdidt(
                    att_partition, available_attributes.copy())
                value_subtree.append(subtree)

            tree.append(value_subtree)
            # print("tree:", tree)

        return tree

    def select_attribute(self, instances, attributes):
        """ recursive function to create a tree from the training data

        Args:
            instances (list of obj): instances used to find entropy of thier attribute values
            attributes (list of str): available attributes to split on
            header (list of str): attribute names

        Returns:
           att_to_split (str): attribute with ideal entropy to split on
        """
        Enew = 1.1  # so the calculated Enew will always be smaller
        att_to_split = ""
        # for each available attribute
        for attr in attributes:
            entropy_list, num_inst_list = [], []
            cur_Enew = 0
            partition = self.partition_instances(
                instances, attr)
            # for each value in the attributes domain
            for val in partition:
                # compute entropy of all instances in val
                entropy = self.compute_entropy(
                    partition[val], len(partition[val]))
                # add to lists for finding Enew
                entropy_list.append(entropy)
                num_inst_list.append(len(partition[val]) / len(instances))
            # compute Enew, the weighted sum of all partition entropies
            cur_Enew = self.compute_Enew(entropy_list, num_inst_list)
            # split on attribute with the smallest Enew
            if cur_Enew < Enew:
                Enew = cur_Enew
                att_to_split = attr
        return att_to_split

    def compute_entropy(self, instances, num_instances):
        """ computes entropy of an attribute

        Args:
            instances (list of obj): instances to compute entropy on
            num_instances (int): number of instances with same attribute value

        Returns:
            entropy (double): entropy for an attribute in the instances
        """
        # least frequent class label, considered the 'negative' label
        negative_class_label = collections.Counter(
            self.y_train).most_common()[-1][0]
        num_falses = 0
        for instance in instances:
            # WILL PROBABLY NEED TO BE CHANGED LATER
            if instance[-1] == negative_class_label:
                num_falses += 1
        num_trues = num_instances - num_falses
        if num_falses == 0 or num_trues == 0:
            return 0
        return ((-num_falses/num_instances) * log2(num_falses/num_instances)) - ((num_trues/num_instances) * log2(num_trues/num_instances))

    def compute_Enew(self, entropy_list, num_inst_list):
        """ determines the most frequent class label in a list of instances

        Args:
            entropy_list (list of int): entropy values for an attribute
            num_inst_list (list of double): proportion of attributes with same value (paarllel to entropy_list)

        Returns:
            label (str): most frequent class label
        """
        Enew = 0
        for i in range(len(entropy_list)):
            Enew += entropy_list[i] * num_inst_list[i]
        return Enew

    def partition_instances(self, instances, split_attribute):
        """ group instances by attribute domain values

        Args:
            instances (list of obj): available instances to be partitioned
            split_attribute (str): attribute to split on
            header (list of str): attribute names

        Returns:
            partitions (dict of list): instances partitioned by attribute values
        """
        partitions = {}  # key (attr value): value (subtable)
        att_index = self.header.index(split_attribute)  # e.g. level -> 0
        att_domain = self.domains[att_index]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def majority_vote_leaf_node(self, instances):
        """ determines the most frequent class label in a list of instances

        Args:
            instances (list of obj): instances to check most frequent class labels

        Returns:
            label (obj): most frequent class label
        """
        class_labels = [instances[i][-1] for i in range(len(instances))]
        class_labels = sorted(class_labels)
        label = max(class_labels, key=class_labels.count)
        return label

    def create_header_and_domains(self):
        """Creates a header list and domain dictionary from self.X_train

        Returns:
            header (list of str): list of attributes
            domains (dict of list): dictionary of attribute indices with possible attribute values
        """
        header = []
        domains = {}
        # create header list and initialize domains
        for i in range(len(self.X_train[0])):
            header.append("att" + str(i))
            domains[i] = []
        # fill domains
        for X in self.X_train:
            for i in range(len(X)):
                # print(domains[i])
                if X[i] not in domains[i]:
                    domains[i].append(X[i])
                domains[i].sort()
        return header, domains


# PREVIOUS CLASSIFIERS BELOW. ABOVE IS NEW TO PA7


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for x in X_test:
            row_idx_distances = []
            for i in range(len(self.X_train)):
                dist = myutils.compute_euclidean_distance(self.X_train[i], x)
                row_idx_distances.append([i, dist])
                # print(dist)
            # sort by closest distances
            row_idx_distances.sort(key=operator.itemgetter(-1))
            # get closest k
            closest_k = row_idx_distances[:self.n_neighbors]
            neig_idx = []
            dist = []
            # get k distances and indices
            for i in range(len(closest_k)):
                neig_idx.append(closest_k[i][0])
                dist.append(closest_k[i][1])
            distances.append(dist)
            neighbor_indices.append(neig_idx)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        _, indices = self.kneighbors(X_test)  # 2 lists of 1 or more lists
        # for each test instance
        for i in range(len(X_test)):
            prediction = []
            # get indices
            idx = indices[i]
            # get predictions from y_train
            for j in range(len(self.y_train)):
                if j in idx:
                    prediction.append(self.y_train[j])
            # use most frequent prediction
            prediction = myutils.get_most_frequent(prediction)
            y_predicted.append(prediction)
        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # ignores X_train
        self.most_common_label = myutils.get_most_frequent(y_train)
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for x in X_test:
            predictions.append(self.most_common_label)
        return predictions


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
                IF y_train IS NONE, CONSTRUCT IT FROM THE LAST COLUMN IN X_train.

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # check if there is a y_train, or if it's atatched to X_train
        # and create list of attribute labels
        if y_train is None:
            y_train = []
            for instance in X_train:
                y_train.append(instance[-1])
                instance.remove(instance[-1])

        # create priors dictionary
        priors = {}
        for y in y_train:
            if y not in priors:
                priors[y] = 1
            elif y in priors:
                priors[y] += 1
        class_counts = dict(sorted(priors.items()))
        # calculate priors
        for val in priors:
            priors[val] = round(priors[val] / len(y_train), 2)
        # set self.priors
        self.priors = dict(sorted(priors.items()))

        # create posteriors
        posteriors = {}
        # go thru X_train, add attributes to posteriors
        for i in range(len(X_train)):
            # add class label to subdictionaries if not in already
            class_label = y_train[i]
            for labels in class_counts:
                if labels not in posteriors:
                    posteriors[labels] = {}
            # go thru each attribute in X_train
            for j in range(len(X_train[i])):
                # j = index, X_train[i][j] = value
                title = "attr" + str(j) + "=" + str(X_train[i][j])
                # if title is not in a class label's subdict, add it
                for labels in class_counts:
                    if title not in posteriors[labels]:
                        posteriors[labels][title] = 0
                # increase count if in the correct class subdict
                if y_train[i] == class_label:
                    posteriors[class_label][title] += 1
            # sort posterior
            posteriors[class_label] = dict(
                sorted(posteriors[class_label].items()))
        # divide values by class label counts to get percentage
        for label in class_counts:
            for attr in posteriors[label]:
                posteriors[label][attr] = round(
                    posteriors[label][attr] / class_counts[label], 2)
        # sort the posteriors
        self.posteriors = dict(sorted(posteriors.items()))
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        # iterate thru each test case
        for x in X_test:
            # for each test case, predict for all prior possibilities
            # parallel lists of class labels and predicted values
            label_list = []
            value_list = []
            # calculate for eact prior, then take largest later
            for label in self.priors:
                cur_pred_vals = 1
                # get all attributes
                for i in range(len(x)):
                    # get name of attribute in posteriors
                    attr_name = "attr" + str(i) + "=" + str(x[i])
                    # multiply all attributes
                    cur_pred_vals *= self.posteriors[label][attr_name]
                # multiply with prior
                cur_pred_vals *= self.priors[label]
                # append label and total value to lists
                label_list.append(label)
                value_list.append(cur_pred_vals)
            # append largest from value_list to y_predicted
            y_predicted.append(label_list[value_list.index(max(value_list))])

        return y_predicted
