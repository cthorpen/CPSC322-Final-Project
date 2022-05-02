from mysklearn.myclassifiers import MyRandomForestClassifier

def test_decision_tree_classifier_fit():
    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
            ["Senior", "Java", "no", "no", "False"],
            ["Senior", "Java", "no", "yes", "False"],
            ["Mid", "Python", "no", "no", "True"],
            ["Junior", "Python", "no", "no", "True"],
            ["Junior", "R", "yes", "no", "True"],
            ["Junior", "R", "yes", "yes", "False"],
            ["Mid", "R", "yes", "yes", "True"],
            ["Senior", "Python", "no", "no", "False"],
            ["Senior", "R", "yes", "no", "True"],
            ["Junior", "Python", "yes", "no", "True"],
            ["Senior", "Python", "yes", "yes", "True"],
            ["Mid", "Python", "no", "yes", "True"],
            ["Mid", "Java", "yes", "no", "True"],
            ["Junior", "Python", "no", "yes", "False"]
        ]

    y_train = []
    for i in range(len(interview_table)):
        y_train.append(interview_table[i].pop(-1))

    interview_forest = MyRandomForestClassifier
