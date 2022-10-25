import sys
from src.algorithms.node import Node


# Here's the Pseudocode (wikipedia):
# Summary
#
#     Calculate the entropy of every attribute a {\displaystyle a} a of the data set S {\displaystyle S} S.
#     Partition ("split") the set S {\displaystyle S} S into subsets using the attribute for which the resulting entropy after splitting is minimized; or, equivalently, information gain is maximum.
#     Make a decision tree node containing that attribute.
#     Recurse on subsets using the remaining attributes.


def compute(df, target_attribute, other_attributes, all_attribute_values):
    assert target_attribute not in other_attributes

    target_vals = df[target_attribute].unique()
    if len(target_vals) == 1:
        return target_vals[0]

    if not other_attributes:
        return df[target_attribute].value_counts().index[0]

    nodes_left = [
        Node(
            o_attr, all_attribute_values[o_attr], target_attribute, all_attribute_values[target_attribute],
            is_indexable=(
                    type(all_attribute_values[o_attr][0]) in [int, float] and len(all_attribute_values[o_attr]) > 30
            )
        )
        for o_attr in other_attributes
    ]

    # Get best attribute based on GAIN
    best_gain = sys.float_info.max * -1
    best_node = None
    for n in nodes_left:

        new_gain = n.get_gain_ratio(df)
        if new_gain > best_gain:
            best_node = n
            best_gain = new_gain

    assert best_node

    children = {}

    # For each child attribute of BA, re-compute
    other_attributes.remove(best_node.attr.name)
    new_other_attributes = other_attributes

    groups = best_node.get_grouping(df)

    assert groups

    best_child = None
    best_len = -1

    for key in groups:
        new_df = groups[key]
        if len(new_df) > best_len or not best_child:
            best_child = key
            best_len = len(new_df)
        if len(new_df) == 0:
            children[key] = df[target_attribute].value_counts().index[0]
        else:
            children[key] = compute(new_df, target_attribute, new_other_attributes, all_attribute_values)

    assert best_child
    children["UNK"] = children[best_child]

    best_node.set_children(children)

    return best_node
