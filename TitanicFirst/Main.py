# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:19:45 2016

@author: Antonio

Main routine, which executes process from data munging to learning process.
"""
import TitanicFirst as tf
import MachineLearningTraining as mlt
import pandas as pd
import csv as csv

#%% Data loading
print ("Loading Data")
analyzedDF = tf.dataTrain()
dataTestDF = tf.dataTest()
print ("Data Loaded")

#%% So, we run adaptations
print ("Running adaptations")
analyzedDF = tf.runDataAdaptations(analyzedDF)
dataTestDF = tf.runDataAdaptations(dataTestDF)
print ("Adaptations executed")

#%% We run training
trainedData = mlt.makeLearningAndPredictions(analyzedDF, dataTestDF)
survivingEstimations = pd.DataFrame({'PassengerId': dataTestDF.PassengerId, 'SurvivedEstimation':pd.Series(trainedData)})

#%% We save results
predictions_file = open("data/myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(survivingEstimations.PassengerId, survivingEstimations.SurvivedEstimation))
predictions_file.close()
print ('Done.')