"""
Collection of util functions for working with parts-of-speech (POS) tags and dependencies.
"""

from collections import Counter


def get_most_common(objects, n):
    return Counter(objects).most_common(n)


def is_pos_tag(token, pos):
    return token.pos_ == pos


def is_dep(token, dep):
    return token.dep_ == dep


def right_is_head_of_left(left, right):
    return left.head == right


def left_is_dep_of_right(left, right, dep):
    return is_dep(left, dep) and right_is_head_of_left(left, right)
