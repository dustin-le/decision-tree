# Dustin Le
# 1001130689

from sys import argv
import numpy as np
import random
from math import log2

def DTL(examples, attributes, default, option):
    if examples.size == 0:
        return default
    
    # If all examples have the same class, then return the class
    elif (len(set(examples[:, -1])) == 1):
        return examples[:, -1][0]
    
    else:
        (best_attribute, best_threshold) = choose_attribute(examples, attributes, option)

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
        return (best_attribute, best_threshold)

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
    return (best_attribute, best_threshold)

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
    
    DTL(examples, attributes, distribution(examples), option)
    
decision_tree(argv[1], argv[2], argv[3], argv[4])
    