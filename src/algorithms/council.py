# All the Algorithms

import sys

import pandas as pd

from src.algorithms.node import Node
from src.algorithms.id3 import compute as id3_compute
from src.algorithms.naivesbayes import compute as nb_compute
from src.algorithms.ann import compute as ann_compute


class Council:
    def __init__(self):
        self.council = []
        self.accuracy = []
        self.target_values = []

    def train(self, t_df, target_attr, other_attr, all_attribute_values, council_func):

        t_df = pd.DataFrame(t_df)
        self.target_values = all_attribute_values[target_attr]
        t_df = t_df.sample(frac=1)
        df_length = len(t_df)
        chunk_size = int(df_length / 8)
        cur_acc = [0, 0, 0]

        for i in range(8):
            testing_df = t_df.copy().iloc[i * chunk_size:(i + 1) * chunk_size, :]
            testing_df_without_answers = testing_df.copy().drop(columns=[target_attr])
            training_df = t_df.copy()
            for i in testing_df.index:
                training_df.drop(i)

            classifiers = [
                func(training_df, target_attr, other_attr.copy(), all_attribute_values.copy()) for func in council_func
            ]

            for j in range(len(classifiers)):
                acc = 0
                for index, row in testing_df_without_answers.iterrows():
                    if classifiers[j].classify(row) == testing_df.loc[index][target_attr]:
                        acc += 1

                cur_acc[j] += acc / len(testing_df)

        self.accuracy = [c_a / 8 for c_a in cur_acc]

        self.council = [
            func(t_df, target_attr, other_attr, all_attribute_values) for func in council_func
        ]

    def classify(self, row):
        retval = [0 for _ in self.target_values]

        for i in range(len(self.council)):
            councilman = self.council[i]
            councilman_acc = self.accuracy[i]

            vote = councilman.classify(row)
            vote_indx = self.target_values.index(vote)

            for j in range(len(retval)):
                if j == vote_indx:
                    retval[j] += councilman_acc - 0.5

        ret_i = 0
        for i in range(len(retval)):
            if retval[i] >= retval[ret_i]:
                ret_i = i

        return self.target_values[ret_i]


def compute(df, target_attribute, other_attributes, all_attribute_values):

    council = Council()
    council_funcs = [
        id3_compute,
        nb_compute,
        ann_compute
    ]
    council.train(df, target_attribute, other_attributes, all_attribute_values, council_funcs)

    return council

