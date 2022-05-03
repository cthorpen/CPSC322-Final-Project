from mysklearn.myclassifiers import MyRandomForestClassifier

def test_random_forest_classifier_fit():
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

    interview_forest = MyRandomForestClassifier()
    interview_forest.fit(interview_table, y_train, interview_header, seed=1, tt_seed=1)

    sample_forest = [
                        ['Attribute', 'lang', 
                            ['Value', 'Java', 
                                ['Leaf', 'False', 3, 9]
                            ], 
                            ['Value', 'Python', 
                                ['Leaf', 'True', 3, 9]
                            ], 
                            ['Value', 'R', 
                                ['Leaf', 'True', 3, 9]
                            ]
                        ], 
                        
                        ['Attribute', 'lang', 
                            ['Value', 'Java', 
                                ['Leaf', 'False', 3, 9]
                            ], 
                            ['Value', 'Python', 
                                ['Leaf', 'True', 3, 9]
                            ], 
                            ['Value', 'R', 
                                ['Leaf', 'True', 3, 9]
                            ]
                        ]
                    ]

    assert sample_forest == interview_forest.trees

def test_random_forest_classifier_predict():
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

    interview_forest = MyRandomForestClassifier()
    interview_forest.fit(interview_table, y_train, interview_header, seed=1, tt_seed=1)

    '''sample_forest = [
                        ['Attribute', 'lang', 
                            ['Value', 'Java', 
                                ['Leaf', 'False', 3, 9]
                            ], 
                            ['Value', 'Python', 
                                ['Leaf', 'True', 3, 9]
                            ], 
                            ['Value', 'R', 
                                ['Leaf', 'True', 3, 9]
                            ]
                        ], 
                        
                        ['Attribute', 'lang', 
                            ['Value', 'Java', 
                                ['Leaf', 'False', 3, 9]
                            ], 
                            ['Value', 'Python', 
                                ['Leaf', 'True', 3, 9]
                            ], 
                            ['Value', 'R', 
                                ['Leaf', 'True', 3, 9]
                            ]
                        ]
                    ]'''

    X_test = [["Senior", "Java", "no", "yes", "False"], ["Mid", "Python", "no", "yes", "True"], ["Junior", "R", "yes", "yes", "False"]]
    y_pred = interview_forest.predict(X_test)

    assert y_pred == ['False', 'True', 'True']