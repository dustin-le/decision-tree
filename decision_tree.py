# Dustin Le
# 1001130689

from sys import argv, setrecursionlimit
import numpy as np
import random
from math import log2

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.id = 1
        self.attribute = None
        self.threshold = None
        self.gain = None

def DTL(examples, attributes, default, option, pruning_thr):
    if len(examples) < pruning_thr:
        tree = Tree()
        tree.attribute = max(default)
        tree.threshold = -1
        tree.gain = 0
        return tree
    
    # If all examples have the same class, then return the class
    elif (len(set(examples[:, -1])) == 1):
        tree = Tree()
        (best_attribute, best_threshold, gain) = choose_attribute(examples, attributes, option)
        tree.attribute = examples[:, -1][0]
        tree.threshold = best_threshold
        tree.gain = gain
        return tree
    
    else:
        (best_attribute, best_threshold, gain) = choose_attribute(examples, attributes, option)
        tree = Tree()
        tree.attribute = best_attribute
        tree.threshold = best_threshold
        tree.gain = gain

        # Appending the data as lists first, and then turning them into np.arrays is more efficient.
        examples_left = []
        examples_right = []
        i = 0
        for val in examples.T[best_attribute]:
            if val < best_threshold:
                examples_left.append(examples[i])
                i += 1
            elif val >= best_threshold:
                examples_right.append(examples[i])
                i += 1
        
        examples_left = np.array(examples_left)
        examples_right = np.array(examples_right)

        tree.left = DTL(examples_left, attributes, distribution(examples), option, pruning_thr)
        tree.right = DTL(examples_right, attributes, distribution(examples), option, pruning_thr)

        return tree


def choose_attribute(examples, attributes, option):
    # Optimized
    if (option == 'optimized'):
        max_gain = best_attribute = best_threshold = -1
        for A in attributes:
            attribute_values = examples.T[A]
            L = min(attribute_values)
            M = max(attribute_values)
            
            for K in range(1, 51):
                threshold = L + K * (M - L) / 51
                gain = information_gain(examples, A, threshold)

                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold
        return (best_attribute, best_threshold, gain)

    # Randomized
    if (option == 'randomized'):
        max_gain = best_attribute = best_threshold = -1
        A = random.choice(attributes)
        attribute_values = examples.T[A]
        L = min(attribute_values)
        M = max(attribute_values)
        
        for K in range(1, 51):
            threshold = L + K * (M - L) / 51
            gain = information_gain(examples, A, threshold)

            if gain > max_gain:
                max_gain = gain
                best_attribute = A
                best_threshold = threshold
    return (best_attribute, best_threshold, gain)

def information_gain(examples, A, threshold):
    # Initialize indices and entropies
    i = left = right = main_entropy = left_entropy = right_entropy = 0

    # Initialize the total number of inputs
    total = examples.shape[0]

    # Keep track of the classes
    classes = examples.T[examples.T.shape[0] - 1]

    # Initialize the class counters for the main, left, and right entropies
    main_count = np.zeros(int(max(examples.T[examples.T.shape[0] - 1]) + 1))
    left_count = np.zeros(int(max(examples.T[examples.T.shape[0] - 1]) + 1))
    right_count = np.zeros(int(max(examples.T[examples.T.shape[0] - 1]) + 1))

    # Count the class alongside calculating where the value belongs based on the threshold
    for val in examples.T[A]:
        main_count[int(classes[i])] += 1
        i += 1
        if val < threshold:
            left_count[int(classes[left])] += 1
            left += 1
        elif val >= threshold:
            right_count[int(classes[right])] += 1
            right += 1

    # Calculate the entropies
    for j in range(main_count.size):
        if (main_count[j] != 0):
            main_entropy -= main_count[j] / total * log2(main_count[j] / total)
        else:
            continue
    for j in range(left_count.size):
        if (left_count[j] != 0):
            left_entropy -= left_count[j] / total * log2(left_count[j] / total)
        else:
            continue
    for j in range(right_count.size):
        if (right_count[j] != 0):
            right_entropy -= right_count[j] / total * log2(right_count[j] / total)
        else:
            continue
    
    # Return the information gain
    return (main_entropy - sum(left_count) / total * left_entropy - sum(right_count) / total * right_entropy)

def distribution(examples):
    count = np.zeros(int(max(examples.T[examples.T.shape[0] - 1]) + 1))
    for val in examples[:, -1]:
        count[int(val)] += 1
    # Return array of probability of i-th class
    return count / sum(count)

def decision_tree(training_file, test_file, option, pruning_thr):
    train = []
    test = []

    with open(training_file, 'r') as training:
        for line in training:
            train.append(line.split())
    
    with open(test_file, 'r') as testing:
        for line in testing:
            test.append(line.split())

    train = np.array(train).astype(np.float)
    test = np.array(test).astype(np.float)

    rows = train.shape[0]
    columns = train.shape[1]

    examples = np.array(train)
    attributes = []
    for i in range(columns - 1):
        attributes.append(i)
    
    tree = DTL(examples, attributes, distribution(examples), option, int(pruning_thr))

    # Assign ID numbers
    current_level = [tree]
    temp = 1
    while current_level:
        next_level = []
        for node in current_level:
            if (node.left):
                temp += 1
                node.left.id = temp
                next_level.append(node.left)
            if (node.right):
                temp += 1
                node.right.id = temp
                next_level.append(node.right)
            current_level = next_level

    # Traverse
    current_level = [tree]
    # while current_level:
    for i in range(5):
        next_level = []
        for node in current_level:
            print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' % (1, node.id, node.attribute, node.threshold, node.gain))
            if (node.left):
                next_level.append(node.left)
            if (node.right):
                next_level.append(node.right)
            current_level = next_level
    # print('ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' % (i+1, e[i], t[i], (e[i] - t[i])**2))

s = 0
nodeID = 1
setrecursionlimit(5000)
decision_tree(argv[1], argv[2], argv[3], argv[4])
    