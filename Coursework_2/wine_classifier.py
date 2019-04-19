#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA

# By default we set figures to be 12"x8" on a 110 dots per inch (DPI) screen
# (adjust DPI if you have a high res screen!)
plt.rc('figure', figsize=(12, 8), dpi=110)
plt.rc('font', size=6)

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']


def calculate_accuracy(gt_labels, pred_labels):
    accuracy_counter = 0
    for i in range(0, len(gt_labels)):
        if gt_labels[i] == pred_labels[i]:
            accuracy_counter += 1
        else:
            continue
    accuracy = (accuracy_counter/len(gt_labels))*100
    return accuracy


def calculate_confusion_matrix(gt_labels, pred_labels):
    k = len(np.unique(gt_labels))
    matrix = np.zeros([k, k])
    for i in range(len(gt_labels)):
        matrix[gt_labels[i]-1][pred_labels[i]-1] += 1
    for i in range(k):
        sum_row = matrix[i][0] + matrix[i][1] + matrix[i][2]
        for j in range(k):
            matrix[i][j] = matrix[i][j]/sum_row
    return matrix


def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.

    Args:
        - matrix: the matrix to be displayed
        - ax: the matplotlib axis where to overlay the plot.
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`.
          If you do not explicitily create a figure, then pass no extra argument.
          In this case the  current axis (i.e. `plt.gca())` will be used
    """
    if ax is None:
        ax = plt.gca()

    # Colour-scheme
    mappable = plt.get_cmap('summer')

    # Plotted matrix with colourbar & text
    img1 = ax.imshow(matrix, cmap=mappable)
    plt.colorbar(img1)
    plt.title("Confusion Matrix")

    size = np.shape(matrix)
    for i in range(size[0]):
        for j in range(size[1]):
            ax.text(i, j, matrix[j][i])
    plt.show()


def feature_selection(train_set, train_labels, **kwargs):
    """
    Returns the two features which from our eyes we have decided look the best to classify the wines

    To view the graphs in which we used to select the features, then uncomment the code below
    """

    # n_features = train_set.shape[1]
    # fig, axarray = plt.subplots(n_features, n_features)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    #
    # colours = np.zeros_like(train_labels, dtype = np.object)
    # colours[train_labels == 1] = CLASS_1_C
    # colours[train_labels == 2] = CLASS_2_C
    # colours[train_labels == 3] = CLASS_3_C
    #
    # for i in range(0,13):
    #     for j in range(0,13):
    #         axarray[i,j].scatter(train_set[:, i],train_set[:, j], c = colours)
    #         axarray[i,j].set_title('Features {} vs {}'.format(i+1,j+1))
    # plt.show()

    return [6, 9]


def knn(train_set, train_labels, test_set, test_labels, k, **kwargs):
    """
    Uses knn classifier to return a predicted set of labels for the test set
    Uncomment the accuracy/confusion code at the bottom if required

    Args:
        -train_set
        -train_labels
        -test_set
        -"test_labels": Only added this to calculate the accuracy of our algorithm
        -k: number of neighbours to calculate the labels
    """

    features = [6, 9]
    train_set_reduced = train_set[:, [features[0], features[1]]]
    test_set_reduced = test_set[:, [features[0], features[1]]]
    knn_predictions = []

    # Loop over all the values in the test set and make a prediction for each one
    for i in range(np.shape(test_set_reduced)[0]):
        distances = []
        k_labels = []
        # Iterate over all the values in the train set and find the euclidian distance to the current value we are predicting for each one
        for j in range(np.shape(train_set_reduced)[0]):
            distances.append([np.sqrt(np.sum(np.square(test_set_reduced[i] - train_set_reduced[j]))), j])

        # Sort the distances
        distances = sorted(distances)

        # Take the value of the labels for the k smallest distances
        for x in range(k):
            k_labels.append(train_labels[distances[x][1]])

        # Function taken from online
        # Finds the most occuring element in a list
        def most_frequent(list):
            return max(set(list), key=list.count)

        knn_predictions.append(most_frequent(k_labels))

    accuracy = calculate_accuracy(test_labels, knn_predictions)
    matrix = calculate_confusion_matrix(test_labels, knn_predictions)
    plot_matrix(matrix)
    print(accuracy)


    return knn_predictions

def probability(x, mean, std):

    return (1 / (math.sqrt(2*math.pi) * std)) * (math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std, 2)))))



def alternative_classifier(train_set, train_labels, test_set, test_labels, **kwargs):
    #Naive Bayes Classifier

    number_of_1 = np.count_nonzero(train_labels == 1)
    number_of_2 = np.count_nonzero(train_labels == 2)
    number_of_3 = np.count_nonzero(train_labels == 3)

    columns = np.size(train_set, 1)

    class_1_data = np.zeros((number_of_1, columns))
    counter_1 = 0

    class_2_data = np.zeros((number_of_2, columns))
    counter_2 = 0

    class_3_data = np.zeros((number_of_3, columns))
    counter_3 = 0



    for i in range(np.size(train_labels)):
        if train_labels[i] == 1:
            new_row = train_set[i, :]
            class_1_data[counter_1] = new_row
            counter_1 = counter_1 + 1

        elif train_labels[i] == 2:
            new_row = train_set[i, :]
            class_2_data[counter_2] = new_row
            counter_2 = counter_2 + 1

        elif train_labels[i] == 3:
            new_row = train_set[i, :]
            class_3_data[counter_3] = new_row
            counter_3 = counter_3 + 1

    feature_1, feature_2 = feature_selection(train_set, train_labels)

    means_1 = class_1_data.mean(axis=0)
    feature_1_class_1_mean = means_1[feature_1]
    feature_2_class_1_mean = means_1[feature_2]

    means_2 = class_2_data.mean(axis=0)
    feature_1_class_2_mean = means_2[feature_1]
    feature_2_class_2_mean = means_2[feature_2]

    means_3 = class_3_data.mean(axis=0)
    feature_1_class_3_mean = means_3[feature_1]
    feature_2_class_3_mean = means_3[feature_2]

    SD_1 = class_1_data.std(axis=0)
    feature_1_class_1_SD = SD_1[feature_1]
    feature_2_class_1_SD = SD_1[feature_2]


    SD_2 = class_2_data.std(axis=0)
    feature_1_class_2_SD = SD_2[feature_1]
    feature_2_class_2_SD = SD_2[feature_2]

    SD_3 = class_3_data.std(axis=0)
    feature_1_class_3_SD = SD_3[feature_1]
    feature_2_class_3_SD = SD_3[feature_2]

    bayes_predictions =[]

    for i in range(np.size(test_labels)):
        sample_feature_1 = test_set[i, feature_1]
        sample_feature_2 = test_set[i, feature_2]

        prob_class_1_feature_1 = probability(sample_feature_1, feature_1_class_1_mean, feature_1_class_1_SD)
        prob_class_1_feature_2 = probability(sample_feature_2, feature_2_class_1_mean, feature_2_class_1_SD)
        total_prob_class_1 = prob_class_1_feature_1*prob_class_1_feature_2

        prob_class_2_feature_1 = probability(sample_feature_1, feature_1_class_2_mean, feature_1_class_2_SD)
        prob_class_2_feature_2 = probability(sample_feature_2, feature_2_class_2_mean, feature_2_class_2_SD)
        total_prob_class_2 = prob_class_2_feature_1 * prob_class_2_feature_2

        prob_class_3_feature_1 = probability(sample_feature_1, feature_1_class_3_mean, feature_1_class_3_SD)
        prob_class_3_feature_2 = probability(sample_feature_2, feature_2_class_3_mean, feature_2_class_3_SD)
        total_prob_class_3 = prob_class_3_feature_1 * prob_class_3_feature_2

        if max(total_prob_class_1,total_prob_class_2,total_prob_class_3) == total_prob_class_1:
            bayes_predictions.append(1)
        elif max(total_prob_class_1,total_prob_class_2,total_prob_class_3) == total_prob_class_2:
            bayes_predictions.append(2)
        elif max(total_prob_class_1, total_prob_class_2, total_prob_class_3) == total_prob_class_3:
            bayes_predictions.append(3)



    bayes_accuracy = calculate_accuracy(test_labels, bayes_predictions)
    print(bayes_accuracy)

    bayes_matrix = calculate_confusion_matrix(test_labels, bayes_predictions)
    plot_matrix(bayes_matrix)

    return bayes_predictions



def knn_three_features(train_set, train_labels, test_set, test_labels, k, **kwargs):
    """
    Uses knn three features classifier to return a predicted set of labels for the test set
    Uncomment the accuracy/confusion code at the bottom if required

    Args:
        -train_set
        -train_labels
        -test_set
        -"test_labels": Only added this to calculate the accuracy of our algorithm
        -k: number of neighbours to calculate the labels
    """

    knn3_features = [6, 9, 12]
    train_set_reduced = train_set[:,[knn3_features[0],knn3_features[1], knn3_features[2]]]
    test_set_reduced = test_set[:,[knn3_features[0],knn3_features[1],knn3_features[2]]]
    predictions = []

    #Loop over all the values in the test set and make a prediction for each one
    for i in range(np.shape(test_set_reduced)[0]):
        distances = []
        klabels = []
        #Iterate over all the values in the train set and find the euclidian distance to the current value we are predicting for each one
        for j in range(np.shape(train_set_reduced)[0]):
            distances.append([np.sqrt(np.sum(np.square(test_set_reduced[i] - train_set_reduced[j]))), j])

        #Sort the distances
        distances = sorted(distances)

        #Take the value of the labels for the k smallest distances
        for x in range(k):
            klabels.append(train_labels[distances[x][1]])

        #Function taken from online
        #Finds the most occuring element in a list
        def most_frequent(List):
            return max(set(List), key = List.count)

        predictions.append(most_frequent(klabels))

    accuracy = calculate_accuracy(test_labels, predictions)
    matrix = calculate_confusion_matrix(test_labels, predictions)
    plot_matrix(matrix)
    print(accuracy)

    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(train_set_reduced,train_labels)
    # pred = knn.predict(test_set_reduced)
    # acc = calculate_accuracy(test_labels,pred)
    # print(acc)

    return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):

    pca = PCA(n_components)
    pca.fit(train_set)
    train_set_reduced = pca.fit_transform(train_set)

    # PLOTS PCA TRAINING SET GRAPH
    # fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    #
    # colours = np.zeros_like(train_labels, dtype = np.object)
    # colours[train_labels == 1] = CLASS_1_C
    # colours[train_labels == 2] = CLASS_2_C
    # colours[train_labels == 3] = CLASS_3_C
    #
    # ax.scatter(training_reduced[:, 0],training_reduced[:, 1], c=colours)
    # ax.set_title('PCA')
    # ax.set_aspect('equal')
    # plt.show()

    test_set_reduced = pca.transform(test_set)
    # test_set_reduced[:, 1] *= -1

    pca_predictions =[]

    # Loop over all the values in the test set and make a prediction for each one
    for i in range(np.shape(test_set_reduced)[0]):
        distances = []
        k_labels = []
        # Iterate over all the values in the train set and find the euclidian distance to the current value we are predicting for each one
        for j in range(np.shape(train_set_reduced)[0]):
            distances.append([np.sqrt(np.sum(np.square(test_set_reduced[i] - train_set_reduced[j]))), j])

        # Sort the distances
        distances = sorted(distances)

        # Take the value of the labels for the k smallest distances
        for x in range(k):
            k_labels.append(train_labels[distances[x][1]])

        # Function taken from online
        # Finds the most occuring element in a list
        def most_frequent(list):
            return max(set(list), key=list.count)

        pca_predictions.append(most_frequent(k_labels))

    accuracy = calculate_accuracy(test_labels, pca_predictions)
    matrix = calculate_confusion_matrix(test_labels, pca_predictions)
    plot_matrix(matrix)
    print(accuracy)

    return pca_predictions




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, test_labels, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set, test_labels)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set,test_labels, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
