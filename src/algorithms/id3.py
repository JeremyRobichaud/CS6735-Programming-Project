import math
import sys
from statistics import mode
from collections import Counter

# Here's the Pseudocode (wikipedia):
# Summary
#
#     Calculate the entropy of every attribute a {\displaystyle a} a of the data set S {\displaystyle S} S.
#     Partition ("split") the set S {\displaystyle S} S into subsets using the attribute for which the resulting entropy after splitting is minimized; or, equivalently, information gain is maximum.
#     Make a decision tree node containing that attribute.
#     Recurse on subsets using the remaining attributes.

# Here's the class object of Node
class Node:
    """:class Node: A class version of a node"""
    def __init__(self, attribute, parent_node=None, parent_value=None):
        """
        :param str attribute: The attribute of this current node. That of which will create the sub-nodes.
        :param Node parent_node: The parent node of this node (None, if it's the root)
        :param str parent_value: The value of the parent node given to this node (None, if it's the root)
        """
        self.attribute = attribute
        self.parent_node = parent_node if parent_node else None
        self.parent_value = parent_value if parent_node else None
        self.parent_attribute = parent_node.get_attribute() if parent_node else None
        self.sub_nodes = []
        self.label = ""
        pass

    def is_leaf(self):
        """
        :return: Whether or not Node is leaf
        :rtype: :bool:
        """
        return len(self.sub_nodes) == 0

    def is_root(self):
        """
        :return: Whether or not Node is root
        :rtype: :bool:
        """
        return not self.parent_node

    def get_attribute(self):
        """
        :return: The attribute of the node
        :rtype: :str:
        """
        return self.attribute

    def get_children(self):
        """
        :return: List of child nodes
        :rtype: :bool:
        """
        return self.sub_nodes

    def add_child(self, new_node):
        """
        :param Node new_node: The new child node
        """
        self.sub_nodes.append(new_node)

    def remove_child(self, old_node):
        """
        param Node old_node: The new child node
        """
        self.sub_nodes.remove(old_node)

    def set_label(self, new_label):
        """
        :param str new_label: The new label
        """
        self.label = new_label


def get_entropy(df, target_attribute, all_target_values):
    all_values = df[target_attribute].value_counts()
    result = 0
    base = len(all_target_values)

    for val in all_values:
        result -= val * math.log(val, base)

    return result


def get_gain(df, target_attribute, all_target_values, calc_attribute):
    calc_vals = df[calc_attribute].unique()
    length_df = len(df)

    gain = get_entropy(df, target_attribute, all_target_values)

    for val in calc_vals:
        temp_df = df.loc[df[calc_attribute] == val]
        length_temp_df = len(temp_df)
        gain -= (length_temp_df/length_df) * get_entropy(temp_df, target_attribute, all_target_values)

    return gain


def compute(df, target_attribute, other_attributes, all_attribute_values):
    target_vals = df[target_attribute].unique()
    if len(target_vals) == 1:
        return target_vals[0]

    if not other_attributes:
        return df[target_attribute].value_counts().index[0]

    best_gain = sys.float_info.max * -1
    best_attr = ''
    for o_attr in other_attributes:

        new_gain = get_gain(df, target_attribute, target_vals, o_attr)
        if new_gain > best_gain:
            best_attr = o_attr
            best_gain = new_gain

    retval = {
        'attr': best_attr,
        'children': {}
    }
    other_attributes.remove(best_attr)
    new_other_attributes = other_attributes
    for val in all_attribute_values[best_attr]:
        new_df = df.loc[df[best_attr] == val]
        if len(new_df) == 0:
            retval['children'][val] = df[target_attribute].value_counts().index[0]
        else:
            temp = compute(new_df, target_attribute, new_other_attributes, all_attribute_values)
            retval['children'][val] = temp

    return retval
