'''
    This program implements a kmeans clustering algorithm.
    This program contains 3 methods:
    The readData(fileName) reads the data into a pandas dataframe.
    The cluster(parsedData = None, iterCount = 0, k = 0, centroids = None) takes a number of iterations and keeps on calculating new centroids by averaging points in specific clusters.
    The calculateSC(clusters, parsedData) takes in the clusters and uses the given formula to calculate the silhouette coefficient.
'''

__author__ = "Manoj Munaganuru"
__version__="03.24.19"

import pandas as pd
from random import seed, randint
import matplotlib.pyplot as plt
import sys

class MyKmeans:
    def readData(self, fileName = ""):
        '''
        :param fileName:
        :return: type DataFrame
        '''
        if type(fileName) is not str:
            return pd.DataFrame()
        if fileName == "":
            return pd.DataFrame()
        try:
            df = pd.read_csv(fileName, header = None)
            df.columns = ["Image ID", "Class Label", "Feature 1", "Feature 2"]
            return df
        except:
            return pd.DataFrame()

    def cluster(self, parsedData = None, iterCount = -1, k = 0, centroids = None):
        '''
        :param parsedData:
        :param iterCount:
        :param k:
        :param centroids:
        :return: list of lists
        '''
        if type(iterCount) is not int or type(k) is not int:
            return [[]]
        if parsedData is None or k == 0 or centroids == [[]] or iterCount <= -1 or k <= -1:
            return [[]]
        image_id = list(parsedData["Image ID"])
        if centroids is None:
            seed(1111)
            centroids = []
            for i in range(k):
                index = randint(0, len(parsedData) - 1)
                if image_id[index] not in centroids:
                    centroids.append(image_id[index])
                else:
                    i -= 1
                    continue
        if k != len(centroids):
            return [[]]
        true_centroids = []
        try:
            for i in range(k):
                true_centroids.append([])
            x_coord = list(parsedData["Feature 1"])
            y_coord = list(parsedData["Feature 2"])
            image_id = list(parsedData["Image ID"])
            for i in range(len(parsedData)):
                min_centroid = 0
                min_value = sys.maxint
                for j in range(len(centroids)):
                    the_index = image_id.index(centroids[j])
                    temp_x = float(x_coord[i] - x_coord[the_index])
                    temp_y = float(y_coord[i] - y_coord[the_index])
                    dist = temp_x**2 + temp_y**2
                    dist = dist ** 0.5
                    if dist < min_value:
                        min_value = dist
                        min_centroid = j
                true_centroids[min_centroid].append(int(image_id[i]))
            the_new_centroids = []
            for i in range(k):
                index = image_id.index(centroids[i])
                the_new_centroids.append([x_coord[index], y_coord[index]])
            for i in range(0, iterCount):
                for j in range(len(true_centroids)):
                    average_x = 0
                    average_y = 0
                    if len(true_centroids[j]) == 0:
                        average_x = sys.maxint
                        average_y = sys.maxint
                        the_new_centroids[j] = [average_x, average_y]
                    else:
                        for l in range(len(true_centroids[j])):
                            index = image_id.index(true_centroids[j][l])
                            average_x += x_coord[index]
                            average_y += y_coord[index]
                        average_x = average_x / len(true_centroids[j])
                        average_y = average_y / len(true_centroids[j])
                        the_new_centroids[j] = [average_x, average_y]
                true_centroids = []
                for j in range(k):
                    true_centroids.append([])
                for l in range(len(parsedData)):
                    min_centroid = 0
                    min_value = sys.maxint
                    for m in range(len(centroids)):
                        temp_x = float(x_coord[l] - the_new_centroids[m][0])
                        temp_y = float(y_coord[l] - the_new_centroids[m][1])
                        dist = temp_x ** 2 + temp_y ** 2
                        dist = dist ** 0.5
                        if dist < min_value:
                            min_value = dist
                            min_centroid = m
                    true_centroids[min_centroid].append(int(image_id[l]))
            return true_centroids
        except:
            return [[]]

    def calculateSC(self, clusters = None, parsedData = None):
        '''
        :param clusters:
        :param parsedData:
        :return: float
        '''
        if clusters is None or parsedData is None or clusters == [[]] or len(clusters) == 1:
            return -1
        try:
            n = len(parsedData)
            x_coord = list(parsedData["Feature 1"])
            y_coord = list(parsedData["Feature 2"])
            image_id = list(parsedData["Image ID"])
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    image_index = clusters[i][j]
                    x_coordinate = x_coord[image_id.index(image_index)]
                    y_coordinate = y_coord[image_id.index(image_index)]
                    clusters[i][j] = [x_coordinate, y_coordinate]

            SC = 0
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    act_b = 0
                    A = 0
                    for k in range(len(clusters[i])):
                        if j == k:
                            continue
                        else:
                            A += float(((clusters[i][j][0] - clusters[i][k][0])**2 +(clusters[i][j][1] - clusters[i][k][1])**2)**0.5)
                    if len(clusters[i]) == 1:
                        A = 0
                    else:
                        A = float(A / (len(clusters[i]) - 1))
                    them_b_vals = []
                    for k in range(len(clusters)):
                        B = 0
                        if k == i:
                            continue
                        else:
                            for l in range(len(clusters[k])):
                                B += float(((clusters[i][j][0] - clusters[k][l][0])**2 + (clusters[i][j][1] - clusters[k][l][1])**2) **0.5)
                            them_b_vals.append(float(B/len(clusters[k])))
                        act_b = min(them_b_vals)
                    SC += float((act_b - A)/max(act_b, A))
            return float(SC / n)
        except:
            return -1
'''
#Project Code

km = MyKmeans()

parsedData = km.readData("digits-embedding.csv")

x_coord_two = []
x_coord_four = []
x_coord_six = []
x_coord_seven = []

y_coord_two = []
y_coord_four = []
y_coord_six = []
y_coord_seven = []

parsedData_set_one = parsedData[(parsedData["Class Label"] == 2) | (parsedData["Class Label"] == 4) | (parsedData["Class Label"] == 6) |(parsedData["Class Label"] == 7)]
parsedData_set_two = parsedData[(parsedData["Class Label"] == 6) |(parsedData["Class Label"] == 7)]

set_one = [2, 4, 6, 7]
set_two = [6, 7]

for i in range(len(parsedData)):
    if parsedData.iloc[i, 1] == 2:
        x_coord_two.append(parsedData.iloc[i, 2])
        y_coord_two.append(parsedData.iloc[i, 3])
    if parsedData.iloc[i, 1] == 4:
        x_coord_four.append(parsedData.iloc[i, 2])
        y_coord_four.append(parsedData.iloc[i, 3])
    if parsedData.iloc[i, 1] == 6:
        x_coord_six.append(parsedData.iloc[i, 2])
        y_coord_six.append(parsedData.iloc[i, 3])
    if parsedData.iloc[i, 1] == 7:
        x_coord_seven.append(parsedData.iloc[i, 2])
        y_coord_seven.append(parsedData.iloc[i, 3])

#Plotting both the subsets of data
plt.scatter(x_coord_two, y_coord_two, s=1, color = "black", label = "Digit 2")
plt.scatter(x_coord_four, y_coord_four, s=1, color = "green", label = "Digit 4")
plt.scatter(x_coord_six, y_coord_six, s=1, color = "orange", label = "Digit 6")
plt.scatter(x_coord_seven, y_coord_seven, s=1, color = "blue", label = "Digit 7")
plt.title("Digits of " + str(set_one))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

plt.scatter(x_coord_six, y_coord_six, s=1, color = "orange", label = "Digit 6")
plt.scatter(x_coord_seven, y_coord_seven, s=1, color = "blue", label = "Digit 7")
plt.title("Digits of " + str(set_two))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

#Running the experiments on the specific subset of data with a specific value of k
scs_for_1_a = []
a_averages = []
b_averages = []
print "k = 2"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_one, 10, 2)
    sc = km.calculateSC(cluster_a, parsedData_set_one)
    print sc
    scs_for_1_a.append(sc)
a_averages.append(float(sum(scs_for_1_a)/len(scs_for_1_a)))
scs_for_1_b = []
print "k = 4"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_one, 10, 4)
    sc = km.calculateSC(cluster_a, parsedData_set_one)
    print sc
    scs_for_1_b.append(sc)
a_averages.append(float(sum(scs_for_1_b)/len(scs_for_1_b)))
scs_for_1_c = []
print "k = 8"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_one, 10, 8)
    sc = km.calculateSC(cluster_a, parsedData_set_one)
    print sc
    scs_for_1_c.append(sc)
a_averages.append(float(sum(scs_for_1_c)/len(scs_for_1_c)))
scs_for_1_d = []
print "k = 16"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_one, 10, 16)
    sc = km.calculateSC(cluster_a, parsedData_set_one)
    print sc
    scs_for_1_d.append(sc)
a_averages.append(float(sum(scs_for_1_d)/len(scs_for_1_d)))
scs_for_2_a = []
print "k = 2"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_two, 10, 2)
    sc = km.calculateSC(cluster_a, parsedData_set_two)
    print sc
    scs_for_2_a.append(sc)
b_averages.append(float(sum(scs_for_2_a)/len(scs_for_2_a)))
scs_for_2_b = []
print "k = 4"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_two, 10, 4)
    sc = km.calculateSC(cluster_a, parsedData_set_two)
    print sc
    scs_for_2_b.append(sc)
b_averages.append(float(sum(scs_for_2_b)/len(scs_for_2_b)))
scs_for_2_c= []
print "k = 8"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_two, 10, 8)
    sc = km.calculateSC(cluster_a, parsedData_set_two)
    print sc
    scs_for_2_c.append(sc)
b_averages.append(float(sum(scs_for_2_c)/len(scs_for_2_c)))
scs_for_2_d = []
print "k = 16"
for i in range(5):
    cluster_a = km.cluster(parsedData_set_two, 10, 16)
    sc = km.calculateSC(cluster_a, parsedData_set_two)
    print sc
    scs_for_2_d.append(sc)
b_averages.append(float(sum(scs_for_2_d)/len(scs_for_2_d)))

plt.scatter([2,4,8,16], a_averages, s=10, color = "red")
plt.title("Silhouette Coefficients of " + str(set_one))
plt.xlabel("K Value")
plt.ylabel("Silhouette Coefficient")
plt.legend()
plt.show()

plt.scatter([2,4,8,16], b_averages, s=10, color = "red")
plt.title("Silhouette Coefficients of " + str(set_two))
plt.xlabel("K Value")
plt.ylabel("Silhouette Coefficient")
plt.legend()
plt.show()
'''