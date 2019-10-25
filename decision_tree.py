# Dustin Le
# 1001130689

from sys import argv
import numpy as np
import random
from math import log2

def choose_attribute(examples, attributes, option):
    # ! I noticed that the min and max of all attributes is 0 and 100, so it's redundantly calculating the threshold.
    # Optimized
    if (option == 'optimized'):
        max_gain = best_attribute = best_threshold = -1
        for A in attributes:
            attribute_values = examples.T[A]
            L = min(attribute_values)
            M = max(attribute_values)

            # * Debug
            print(L, M)
            
            for K in range(1, 51):
                threshold = L + K * (M - L) / 51
                gain = information_gain(examples, A, threshold)

    #             if gain > max_gain:
    #                 max_gain = gain
    #                 best_attribute = A
    #                 best_threshold = threshold
    #     return (best_attribute, best_threshold)

    # # Randomized
    # if (option == 'randomized'):
    #     max_gain = best_attribute = best_threshold = -1
    #     A = random.choice(attributes)
    #     attribute_values = examples.T[A]
    #     L = min(attribute_values)
    #     M = max(attribute_values)
        
    #     for K in range(1, 51):
    #         threshold = L + K * (M - L) / 51
    #         gain = information_gain(examples, A, threshold)

    #         if gain > max_gain:
    #             max_gain = gain
    #             best_attribute = A
    #             best_threshold = threshold
    # return (best_attribute, best_threshold)

def DTL(examples, attributes, default):
    if examples == []:
        return default
    
    # else if all examples have the same class, return class

    # else:

# ! How do you do information gain with no entropies for the child nodes?
def information_gain(examples, A, threshold):
    left = right = 0
    for val in examples.T[A]:
        if val < threshold:
            left += 1
        elif val >= threshold:
            right += 1
    
    main_entropy = -(left / (left + right) * log2(left / (left + right))) - -(right / (left + right) * log2(right / (left + right)))


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

    # ! The instructions say not to include the last column (class labels), but the pseudocode (slide 27) says we should.
    examples = np.array(train)
    attributes = []
    for i in range(columns - 1):
        attributes.append(i)
    
    # * Debug - Remove once done
    choose_attribute(examples, attributes, option)
    

    
decision_tree(argv[1], argv[2], argv[3], argv[4])
    