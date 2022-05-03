import pickle

# create header and data for our prediction (probably a tree)
header = ["Drug", "Age", "Condition", "Season",
          "EaseofUse", "Satisfaction", "Sex"]
# header = ["att0", "att1", "att2", "att3", "att4", "att5", "att6"]

# done with Random Forest where N=15, F=3, M=5.
# accuracy = 0.33, error rate = 0.67
# recall = 0.4
# precision = 0.44
# F1 = 0.4
data = [
    # tree 1
    ['Attribute', 'EaseofUse',
        ['Value', 'VE',
            ['Attribute', 'Sex',
                ['Value', 'Female',
                    ['Leaf', 'VE', 241, 302]
                 ],
                ['Value', 'Male',
                    ['Leaf', 'VE', 61, 302]
                 ]
             ]
         ],
        ['Value', 'SE',
            ['Attribute', 'Sex',
                ['Value', 'Female',
                    ['Leaf', 'SE', 9, 18]
                 ],
                ['Value', 'Male',
                    ['Leaf', 'E', 9, 18]
                 ]
             ]
         ],
        ['Value', 'E',
            ['Attribute', 'Sex',
                ['Value', 'Female',
                    ['Leaf', 'E', 117, 144]
                 ],
                ['Value', 'Male',
                    ['Leaf', 'E', 27, 144]
                 ]
             ]
         ],
        ['Value', 'NE',
            ['Attribute', 'Sex',
                ['Value', 'Female',
                    ['Leaf', 'NE', 15, 20]
                 ],
             ['Value', 'Male',
                    ['Leaf', 'NE', 5, 20]
              ]
             ]
         ],
        ['Value', 'ME',
            ['Attribute', 'Sex',
                ['Value', 'Female',
                    ['Leaf', 'E', 49, 62]
                 ],
                ['Value', 'Male',
                    ['Leaf', 'ME', 13, 62]
                 ]
             ]
         ]
     ],
    # tree 2
    ['Attribute', 'Satisfaction',
        ['Value', 'E',
            ['Attribute', 'EaseofUse',
                ['Value', 'VE',
                    ['Leaf', 'E', 77, 176]
                 ],
                ['Value', 'SE',
                    ['Leaf', 'E', 3, 176]
                 ],
                ['Value', 'E',
                    ['Leaf', 'E', 76, 176]
                 ],
                ['Value', 'NE',
                    ['Leaf', 'VE', 1, 176]
                 ],
                ['Value', 'ME',
                    ['Leaf', 'E', 19, 176]
                 ]
             ]
         ],
        ['Value', 'VE',
            ['Attribute', 'EaseofUse',
                ['Value', 'VE',
                    ['Leaf', 'VE', 185, 209]
                 ],
                ['Value', 'SE',
                    ['Leaf', 'VE', 1, 209]
                 ],
                ['Value', 'E',
                    ['Leaf', 'E', 17, 209]
                 ],
                ['Value', 'NE',
                    ['Leaf', 'ME', 1, 209]
                 ],
                ['Value', 'ME',
                    ['Leaf', 'E', 5, 209]
                 ]
             ]
         ],
        ['Value', 'SE',
            ['Attribute', 'EaseofUse',
                ['Value', 'VE',
                    ['Leaf', 'SE', 4, 33]
                 ],
                ['Value', 'SE',
                    ['Leaf', 'SE', 3, 33]
                 ],
                ['Value', 'E',
                    ['Leaf', 'ME', 10, 33]
                 ],
                ['Value', 'NE',
                    ['Leaf', 'ME', 2, 33]
                 ],
                ['Value', 'ME',
                    ['Leaf', 'ME', 14, 33]
                 ]
             ]
         ],
        ['Value', 'ME',
            ['Attribute', 'EaseofUse',
                ['Value', 'VE',
                    ['Leaf', 'ME', 29, 88]
                 ],
                ['Value', 'SE',
                    ['Leaf', 'ME', 4, 88]
                 ],
                ['Value', 'E',
                    ['Leaf', 'ME', 34, 88]
                 ],
                ['Value', 'NE',
                    ['Leaf', 'E', 1, 88]
                 ],
                ['Value', 'ME',
                    ['Leaf', 'ME', 20, 88]
                 ]
             ]
         ],
        ['Value', 'NE',
            ['Attribute', 'EaseofUse',
                ['Value', 'VE',
                    ['Leaf', 'NE', 7, 40]
                 ],
                ['Value', 'SE',
                    ['Leaf', 'NE', 7, 40]
                 ],
                ['Value', 'E',
                    ['Leaf', 'NE', 7, 40]
                 ],
                ['Value', 'NE',
                    ['Leaf', 'NE', 15, 40]
                 ],
                ['Value', 'ME',
                    ['Leaf', 'NE', 4, 40]
                 ]
             ]
         ]
     ],
    # tree 3
    ['Attribute', 'EaseofUse',
        ['Value', 'VE',
            ['Attribute', 'Age',
                ['Value', '19-24',
                    ['Leaf', 'VE', 77, 302]
                 ],
                ['Value', '25-34',
                    ['Leaf', 'VE', 177, 302]
                 ],
                ['Value', '7-12',
                    ['Leaf', 'ME', 6, 302]
                 ],
                ['Value', '13-18',
                    ['Leaf', 'VE', 15, 302]
                 ],
                ['Value', '3-6',
                    ['Leaf', 'VE', 7, 302]
                 ],
                ['Value', '0-2',
                    ['Leaf', 'E', 6, 302]
                 ],
                ['Value', '55-64',
                    ['Leaf', 'E', 1, 302]
                 ],
                ['Value', '45-54',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '35-44',
                    ['Leaf', 'NE', 2, 302]
                 ],
                ['Value', '75 or over',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '65-74',
                    ['Leaf', 'VE', 3, 302]
                 ]
             ]
         ],
        ['Value', 'SE',
            ['Leaf', 'SE', 18, 18]
         ],
        ['Value', 'E',
            ['Leaf', 'E', 144, 144]
         ],
        ['Value', 'NE',
            ['Leaf', 'NE', 20, 20]
         ],
        ['Value', 'ME',
            ['Leaf', 'ME', 62, 62]
         ]
     ],
    # tree 4
    ['Attribute', 'EaseofUse',
        ['Value', 'VE',
            ['Attribute', 'Age',
                ['Value', '19-24',
                    ['Leaf', 'VE', 77, 302]
                 ],
                ['Value', '25-34',
                    ['Leaf', 'VE', 177, 302]
                 ],
                ['Value', '7-12',
                    ['Leaf', 'ME', 6, 302]
                 ],
                ['Value', '13-18',
                    ['Leaf', 'VE', 15, 302]
                 ],
                ['Value', '3-6',
                    ['Leaf', 'VE', 7, 302]
                 ],
                ['Value', '0-2',
                    ['Leaf', 'E', 6, 302]
                 ],
                ['Value', '55-64',
                    ['Leaf', 'E', 1, 302]
                 ],
                ['Value', '45-54',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '35-44',
                    ['Leaf', 'NE', 2, 302]
                 ],
                ['Value', '75 or over',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '65-74',
                    ['Leaf', 'VE', 3, 302]
                 ]
             ]
         ],
        ['Value', 'SE',
            ['Leaf', 'SE', 18, 18]
         ],
        ['Value', 'E',
            ['Leaf', 'E', 144, 144]
         ],
        ['Value', 'NE',
            ['Leaf', 'NE', 20, 20]
         ],
        ['Value', 'ME',
            ['Leaf', 'ME', 62, 62]
         ]
     ],
    # tree 5
    ['Attribute', 'EaseofUse',
        ['Value', 'VE',
            ['Attribute', 'Age',
                ['Value', '19-24',
                    ['Leaf', 'VE', 77, 302]
                 ],
                ['Value', '25-34',
                    ['Leaf', 'VE', 177, 302]
                 ],
                ['Value', '7-12',
                    ['Leaf', 'ME', 6, 302]
                 ],
                ['Value', '13-18',
                    ['Leaf', 'VE', 15, 302]
                 ],
                ['Value', '3-6',
                    ['Leaf', 'VE', 7, 302]
                 ],
                ['Value', '0-2',
                    ['Leaf', 'E', 6, 302]
                 ],
                ['Value', '55-64',
                    ['Leaf', 'E', 1, 302]
                 ],
                ['Value', '45-54',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '35-44',
                    ['Leaf', 'NE', 2, 302]
                 ],
                ['Value', '75 or over',
                    ['Leaf', 'VE', 4, 302]
                 ],
                ['Value', '65-74',
                    ['Leaf', 'VE', 3, 302]
                 ]
             ]
         ],
        ['Value', 'SE',
            ['Leaf', 'SE', 18, 18]
         ],
        ['Value', 'E',
            ['Leaf', 'E', 144, 144]
         ],
        ['Value', 'NE',
            ['Leaf', 'NE', 20, 20]
         ],
        ['Value', 'ME',
            ['Leaf', 'ME', 62, 62]
         ]
     ]]


# fix this later when we have a Model
packaged_obj = [header, data]
outfile = open("drug.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()
