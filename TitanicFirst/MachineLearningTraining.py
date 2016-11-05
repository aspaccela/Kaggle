# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:22:17 2016

@author: Antonio

Python code for learning from data.
"""

from sklearn import tree

def sliceAsFeatures(dataDF):
    #Se queda solo con las features interesantes, cambiando sex por 1/0
    dataDFAux = dataDF[['Pclass','Age', 'SibSp', 'Parch', 'Fare', 'Sex']]
    return dataDFAux
    
def sliceAsValues(dataDF):
    #Se queda solo con la serie de Survived
    return dataDF[['Survived']]

def makeDecissionTreeLearning(dataDF):
    #Devuelve el DecisionTreeRegressor entrenado
    dtr=tree.Ra()
    dtrValues= sliceAsValues(dataDF)
    dtrFeatures= sliceAsFeatures(dataDF)
    dtr = dtr.fit(dtrFeatures,dtrValues)
    return dtr
    
def predictDecisionTreeLearning(dtRegressor, testDF):
    #Hace la predicción con los datos de Test
    return dtRegressor.predict(testDF)
    
def makeLearningAndPredictions(trainDF, testDF):
    #Realiza el entreno y la predicción
    dtr = makeDecissionTreeLearning(trainDF)
    return predictDecisionTreeLearning(dtr, sliceAsFeatures(testDF))
    
#makeLearningAndPredictions(tf.analyzedDF, tf.dataTestDF)