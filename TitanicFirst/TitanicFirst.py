#%% Celda principal
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 23:30:15 2016

@author: Antonio
"""

import pandas as pd

def dataTrain ():
    dataDF = pd.read_csv("./data/train.csv") 
    return dataDF

def dataTest ():
    dataDF = pd.read_csv("./data/test.csv") 
    return dataDF

#%% Functions to make data adaptations.
def runDataAdaptations(dataDF):
    sexSerieAux = dataDF.Sex.map(mapSex)
    dataDF.Sex = sexSerieAux
    return dropNonValuableFields(fillEmptyValuesAge(fillEmptyValuesEmbarked(fillEmptyValuesFare(dataDF))))

#Fill Empty Values.
def fillEmptyValuesAge(dataDF):
    # Campo Age
    dataDF.Age = dataDF.Age.fillna(dataDF.Age.median())
    return dataDF

def fillEmptyValuesEmbarked(dataDF):
    # Campo Age
    dataDF.Embarked = dataDF.Embarked.fillna(dataDF.Embarked.mode())
    return dataDF

def fillEmptyValuesFare(dataDF):
    # Campo Age
    # TODO. Hay que dar un valor Fare como la media de la clase.
    dataDF.Fare = dataDF.Fare.fillna(1)
    return dataDF

#Drop non-util fields function from the dataset 
def dropNonValuableFields(dataDF):
    # Field Name. Drop.
    # Field Cabin. Drop.
    # Field Ticket. Drop
    return dataDF.drop(['Cabin','Name', 'Ticket'], axis=1)

def mapSex(sex):
    #Función que dado el sexo devuelve 1=male, 0=otros
    if sex=='male':
        value = 1
    else:
        value = 0
    return value 