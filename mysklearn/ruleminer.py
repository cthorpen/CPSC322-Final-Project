import itertools
from tabulate import tabulate
from mysklearn import myutils


class MyAssociationRuleMiner:
    """Represents an association rule miner.

    Attributes:
        minsup(float): The minimum support value to use when computing supported itemsets
        minconf(float): The minimum confidence value to use when generating rules
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        rules(list of dict): The generated rules

    Notes:
        Implements the apriori algorithm
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, minsup=0.25, minconf=0.8):
        """Initializer for MyAssociationRuleMiner.

        Args:
            minsup(float): The minimum support value to use when computing supported itemsets
                (0.25 if a value is not provided and the default minsup should be used)
            minconf(float): The minimum confidence value to use when generating rules
                (0.8 if a value is not provided and the default minconf should be used)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.X_train = None
        self.rules = None

    def fit(self, X_train):
        """Fits an association rule miner to X_train using the Apriori algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)

        Notes:
            Store the list of generated association rules in the rules attribute
            If X_train represents a non-market basket analysis dataset, then:
                Attribute labels should be prepended to attribute values in X_train
                    before fit() is called (e.g. "att=val", ...).
                Make sure a rule does not include the same attribute more than once
        """
        self.X_train = X_train
        self.rules = []

        all_Ls = []

        # 1. create Lsub1 = set of supported itemsets of cardinality (size) 1
        L = self.create_singleton_set()  # set of lists
        # 2. set k = 2
        k = 2
        # 3. while Lsubk-1 != 0
        while (len(L) != 0):
            # 4. create Csubk from Lsubk-1
            Csubk = self.prune_and_join(L, k)
            # 5. prune all itemsets in Csubk that are not supported, to create Lsubk
            L = self.check_support(Csubk)  # should be just L ?
            # keep all L's
            if len(L) != 0:
                for l in L:
                    all_Ls.append(l)
            # 6. increase k by 1
            k += 1
        # 7. the set of all supported itemsets with at least two members is Lsub2 U ... U Lsubk-2
        self.create_rules(all_Ls)
        # calculate interestingness
        for rule in self.rules:
            myutils.compute_rule_interestingness(rule, self.X_train)

        pass  # TODO: fix this

    def print_association_rules(self):
        """Prints the association rules in the format "IF val AND ... THEN val AND...", one rule on each line.

        Notes:
            Each rule's output should include an identifying number, the rule, the rule's support,
            the rule's confidence, and the rule's lift
            Consider using the tabulate library to help with this: https://pypi.org/project/tabulate/
        """
        rule_headers = ["#", "association rule",
                        "support", "confidence", "lift"]
        rule_table = []
        count = 1

        for rule in self.rules:
            cur_rule = ""
            # print lhs
            cur_rule += "IF "
            for lhs in rule["lhs"]:
                cur_rule += lhs
                if rule["lhs"].index(lhs) != len(rule["lhs"]) - 1:
                    cur_rule += " AND "
            # print rhs
            cur_rule += " THEN "
            for rhs in rule["rhs"]:
                cur_rule += rhs
                if rule["rhs"].index(rhs) != len(rule["rhs"]) - 1:
                    cur_rule += " AND "
            rule_table.append([count, cur_rule, round(rule["support"], 2),
                               round(rule["confidence"], 2), round(rule["completeness"], 2)])
            count += 1
        # pretty print rules
        print(tabulate(rule_table, rule_headers))
        pass  # TODO: fix this

    def create_rules(self, itemsets):
        # split each item in itemsets into LHS and RHS
        for sets in itemsets:
            # split 'sets' into subsets
            powerset = []
            for i in range(1, len(sets)):
                powerset.extend(itertools.combinations(sets, i))
            for i in range(len(powerset)):
                powerset[i] = list(powerset[i])

            # separate into lhs and rhs
            for i in range(len(powerset)):
                # get rhs
                rhs = powerset[i]
                for j in range(i, len(powerset)):
                    lhs = powerset[j]
                    # find corresponding lhs
                    check = any(item in rhs for item in lhs)
                    if check == False:
                        if len(sets) > 2:
                            if len(lhs) != len(rhs):
                                self.compare_sides(lhs, rhs)
                        else:
                            self.compare_sides(lhs, rhs)
        return

    def compare_sides(self, lhs, rhs):
        for _ in range(0, 2):
            # compute confidence for lhs vs rhs
            confidence, support = self.compute_confidence_support(lhs, rhs)
            if (confidence >= self.minconf) and (support >= self.minsup):
                self.rules.append({"lhs": lhs, "rhs": rhs})
            # flip sides, so lhs=rhs and rhs=lhs and compute confidence again
            temp = lhs
            lhs = rhs
            rhs = temp
        pass

    def compute_confidence_support(self, lhs, rhs):
        # calculate Nboth
        Nboth = Nleft = 0
        for row in self.X_train:
            Nleft += myutils.check_row_match(lhs, row)
            Nboth += myutils.check_row_match(lhs + rhs, row)
        try:
            confidence = (Nboth / Nleft)
        except ZeroDivisionError:
            confidence = 0.0
        try:
            support = (Nboth / len(self.X_train))
        except ZeroDivisionError:
            support = 0.0
        return confidence, support

    def prune_and_join(self, L, k):
        # generates Csubk from Lsubk-1
        Csubk = []
        if k == 2:
            # 1st ITERATION
            elements = []
            for subset in L:
                for item in subset:
                    elements.append(item)
            Csubk.extend(itertools.combinations(elements, k))
            for i in range(len(Csubk)):
                Csubk[i] = list(Csubk[i])
            # 4b. prune step
        # for each member c of Csubk in turn
            for sets in Csubk:
                # examine all subsets of c with k-1 elements
                for item in sets:
                    # delete c from Csubk if any of the subsets is not a member of Lsubk-1
                    if [item] not in L:
                        Csubk.remove(sets)
        # 4a. join step
        # compare each member of Lsubk-1, say A, w/ every other member, say B, in turn.
        # if the first k-2 items in A and B (i.e. all but the rightmost elements of the first two itemsets)
        # are identical, place set A U B into Csubk.
        elif k - 2 > 0:
            leftmost_elem_num = k - 2
            # join to create subsets of size k
            # iterate thru L
            for i in range(len(L)):
                ct = i
                # iterate thru subsets in L
                for j in range(i+1, len(L)):
                    # check first k-2 items in each subset match
                    leftmost_match = True
                    for elem in range(leftmost_elem_num):
                        if L[ct][elem] != L[j][elem]:
                            leftmost_match = False
                            break
                    # first elements in subset match
                    if leftmost_match == True:
                        subset = set(L[ct]).union(set(L[j]))
                        Csubk.append(sorted(list(subset)))
           # 4b. prune step
            # for each member c of Csubk in turn
            for sets in Csubk:
                # create all subsets of 'sets' with k-1 elements
                subsets = []
                subsets.extend(itertools.combinations(sets, k-1))
                # make a list of list again
                for i in range(len(subsets)):
                    subsets[i] = list(subsets[i])
                # delete set from Csubk if any subset is not a member of L
                for sub in subsets:
                    if sub not in L:
                        Csubk.remove(sets)
                        break
        return Csubk

    def check_support(self, Csubk):  # step 5
        Ntotal = len(self.X_train)
        # check each set in Csubk
        for sets in Csubk:
            # check each item in the sets
            set_items = []
            for item in sets:
                set_items.append(item)
            # check if all elements in set_items are in any rows in X_train
            Nboth = 0
            for row in self.X_train:
                result = all(elem in row for elem in set_items)
                if result:
                    Nboth += 1
            if (Nboth / Ntotal) < self.minsup:
                Csubk.remove(sets)
        return Csubk

    def create_singleton_set(self):
        # create empty list
        singleton = []
        # fill list
        for row in self.X_train:
            for item in row:
                if item not in singleton:
                    singleton.append(item)
        # check support for singleton list
        Nboth = len(self.X_train)
        for single in singleton:
            Ntotal = 0
            for row in self.X_train:
                if single in row:
                    Ntotal += 1
            if (Ntotal / Nboth) < self.minsup:
                singleton.remove(single)
        # convert to powerset
        powerset = []
        powerset.extend(itertools.combinations(singleton, 1))
        for i in range(len(powerset)):
            powerset[i] = list(powerset[i])
        return sorted(powerset)
