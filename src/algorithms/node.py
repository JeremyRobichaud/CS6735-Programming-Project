# This is a Node Class
import math


class Node:
    class _Attribute:
        def __init__(self, name, values):
            self.name = name
            self.values = values

    def __init__(self, attribute_name, attribute_values, target_name, target_values, is_indexable=True):
        attribute_values = list(attribute_values)
        if "?" in attribute_values:
            attribute_values.remove("?")
        self.attr = self._Attribute(attribute_name, attribute_values)
        self._target_attr = self._Attribute(target_name, target_values)
        self.is_indexable = is_indexable and len(target_values) > 2 and type(target_values[0]) in [int, float]
        self.children = {}

    def classify(self, row):
        assert self.children
        target_value = row[self.attr.name]
        child_value = target_value

        # There's 4 possibilities:
        # If Indexible:
        #   '<=' or '>'
        # If not Indexible:
        #   target_value or UNK
        if self.is_indexable:
            if type(target_value) not in [int, float]:
                target_value = float(target_value)
            if target_value <= self._index:
                child_value = "<="
            else:
                child_value = ">"
        elif target_value not in self.children.keys():
            child_value = "UNK"

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
        return (1/2) * math.log(
            (1-self.get_total_error(df, weight_value)) / (self.get_total_error(df, weight_value))
        )

    def _get_index(self, df):
        # I used GINI indexing here, but we can use whatever
        if not self.is_indexable:
            return

        best_val = ""
        best_gain = None
        # The index will be the point of which it has the greatest gain as a separation point
        for val in self.attr.values:
            gain = self.get_gain(df.loc[df[self.attr.name] <= val])
            gain += self.get_gain(df.loc[df[self.attr.name] > val])
            if gain > best_gain or not best_gain:
                best_gain = gain
                best_val = val

        self._index = best_val

    def set_children(self, children):
        
        if self.is_indexable:
            assert "<=" in children.keys()
            assert ">" in children.keys()
            assert len(children) == 2
            self.children = children
            return

        for v in self.attr.values:
            assert v in children.keys()

        assert "UNK" in children.keys()
        assert len(children) == len(self.attr.values) + 1

        self.children = children

    def get_grouping(self, df):
        retval = {}
        if self.is_indexable:
            retval["<="] = df.loc[df[self.attr.name] <= self._index]
            retval[">"] = df.loc[df[self.attr.name] > self._index]
            return retval

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
