# This is a Node Class
import math


class Node:
    class _Attribute:
        def __init__(self, name, values):
            self.name = name
            self.values = values

    def __init__(self, attribute_name, attribute_values, target_name, target_values):
        attribute_values = list(attribute_values)
        self.attr = self._Attribute(attribute_name, attribute_values)
        self._target_attr = self._Attribute(target_name, target_values)
        self.children = {}

    def classify(self, row):
        assert self.children
        target_value = row[self.attr.name]
        child_value = target_value

        if type(self.children[child_value]) == Node:
            return self.children[child_value].classify(row)
        return self.children[child_value]

    def get_total_error(self, df, weight_value):
        total_error = 0

        for index, row in df.iterrows():
            if self.classify(row) != row[self._target_attr.name]:
                total_error += row[weight_value]

        # Small constant is added so that it doesn't given an error
        k = 0.000000001
        return total_error + k

    def get_amount_of_say(self, df, weight_value):
        total_error = self.get_total_error(df, weight_value)
        return 0.5 * math.log((1-total_error) / total_error)

    def set_children(self, children):

        for v in self.attr.values:
            assert v in children.keys()

        assert len(children) == len(self.attr.values)

        self.children = children

    def get_grouping(self, df):
        retval = {}

        for val in self.attr.values:
            retval[val] = df.loc[df[self.attr.name] == val]

        return retval

    def _get_entropy(self, df):
        all_values = df[self._target_attr.name].value_counts()
        result = 0
        base = len(self._target_attr.values)

        for length_val in all_values:
            result -= length_val * math.log(length_val, base)

        return result

    def get_gain(self, df):
        length_df = len(df)

        gain = self._get_entropy(df)

        for temp_df in self.get_grouping(df).values():
            length_temp_df = len(temp_df)
            gain -= (length_temp_df / length_df) * self._get_entropy(temp_df)

        return gain

    def _get_split_info(self, df):
        all_values = df[self._target_attr.name].value_counts()
        result = 0
        length_df = len(df)

        for length_val in all_values:
            result -= (length_val/length_df) * math.log((length_val/length_df), 2)

        return result

    def get_gain_ratio(self, df):
        gain = self.get_gain(df)
        si = self._get_split_info(df)
        return gain / si
