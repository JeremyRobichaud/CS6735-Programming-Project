import math
import random

from src.algorithms.node import Node
from src.algorithms.id3 import ID3_process

import pandas as pd


def _get_stumps(df, target_attribute, other_attributes, all_attribute_values):
    stumps = []
    cur_df = df.copy()
    for _ in range(10):
        cur_stump = ID3_process(cur_df.copy(), target_attribute, other_attributes.copy(), all_attribute_values)

        if isinstance(cur_stump, str):
            continue

        # TODO: Do I need to do this? Not Sure
        # other_attributes.remove(cur_stump.attr.name)

        cur_weight = 1 / len(cur_df)
        temp_df = cur_df.copy()
        temp_df['cur_weight'] = [cur_weight for _ in range(len(temp_df))]

        cur_amount_of_say = cur_stump.get_amount_of_say(temp_df, 'cur_weight')
        new_weights = []

        stumps.append([cur_stump, cur_amount_of_say])

        for index, row in cur_df.copy().iterrows():
            if cur_stump.classify(row) != row[target_attribute]:
                # incorrect_weight = sample_weight * e^cur_amount_of_say
                new_weights.append(cur_weight * math.exp(cur_amount_of_say))
            else:
                # correct_weight = sample_weight * e^-cur_amount_of_say
                new_weights.append(cur_weight * math.exp(-1 * cur_amount_of_say))

        # Normalize new Weights
        sum_nw = sum(new_weights)
        new_weights = [nw / sum_nw for nw in new_weights]

        # We will use weighted distributions over GINI Indexing
        new_df = cur_df.copy()
        new_df.drop(new_df.index[:], inplace=True)

        while len(new_df) < len(cur_df) or len(list(new_df[target_attribute].unique())) < 2:
            generated_num = random.random()
            cur_sum = 0
            for j in range(len(new_weights)):
                if generated_num < cur_sum + new_weights[j]:
                    row = dict(cur_df.iloc[j])
                    new_row = pd.DataFrame(
                        row, index=[0]
                    )
                    new_df = pd.concat([new_row, new_df.loc[:]]).reset_index(drop=True)
                    break
                cur_sum += new_weights[j]
        cur_df = new_df.copy()

    return stumps


class Adaboost:
    def __init__(self, stumps, target_values):
        for s in stumps:
            assert len(s) == 2
            assert isinstance(s[0], Node)
            assert isinstance(s[1], type(0.1))
        self._stumps = stumps
        self._target_values = target_values

    def classify(self, row):
        answers = [0 for _ in self._target_values]
        for stump in self._stumps:
            answer = stump[0].classify(row)
            answers[list(self._target_values).index(answer)] = stump[1]

        return list(self._target_values)[answers.index(max(answers))]


def compute(df, target_attribute, other_attributes, all_attribute_values):
    stumps = _get_stumps(df, target_attribute, other_attributes, all_attribute_values)

    return Adaboost(stumps, all_attribute_values[target_attribute])
