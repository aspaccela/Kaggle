# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:12:59 2016

@author: Antonio
"""
import Main as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Pruebas acerca de si Embarked es un campo implicado en el resultado final.
totalPassengerByEmbarkedDF= m.analyzedDF.groupby(m.analyzedDF.Embarked).count().filter(items=['Survived'])
print (totalPassengerByEmbarkedDF)

survivedPassengerByEmbarkedDF=m.analyzedDF[m.analyzedDF.Survived==1].groupby(m.analyzedDF.Embarked).count().filter(items=['Survived'])
print (survivedPassengerByEmbarkedDF)

percentagePassengerByEmbarkedDF=survivedPassengerByEmbarkedDF/totalPassengerByEmbarkedDF
print(percentagePassengerByEmbarkedDF)    

## Parece que el puerto de embarque sí afecta al resultado final


#%% Tests about if Cabin field can be related to the final result.
totalPassengerByCabinDF=m.analyzedDF.groupby(m.analyzedDF.Cabin).count().filter(items=['Survived'])
print (totalPassengerByCabinDF)

survivedPassengerByCabinDF=m.analyzedDF[m.analyzedDF.Survived==1].groupby(m.analyzedDF.Cabin).count().filter(items=['Survived'])
print (survivedPassengerByCabinDF)

percentagePassengerByCabinDF=survivedPassengerByCabinDF/totalPassengerByCabinDF
print(percentagePassengerByCabinDF)    
# Let's see more common Cabins
m.analyzedDF.Cabin.describe()
m.analyzedDF[pd.isnull(m.analyzedDF.Cabin)].count()
# We can establish that the more common cabin is the null cabin. That's why we are going to exclude the field from the analysis


#%% Analysis about ticket field
m.analyzedDF.Ticket.describe()
# 681 unique values, 7 on a maximum frequency. We can avoid its use.

#%% Compare ticket fare with surviving rate

#Search and partition data by tickect fare to look after Survived people.
#indicesFare=pd.cut(m.analyzedDF.Fare, 6)
#m.analyzedDF[m.analyzedDF.Survived==1].groupby(indicesFare)

#Separate between survived and not and see for differences
fareNotSurvivedSr=m.analyzedDF[m.analyzedDF.Survived==0].Fare
fareSurvivedSr=m.analyzedDF[m.analyzedDF.Survived==1].Fare
print("Fare Survived")
print(fareSurvivedSr.describe())
print ("Fare not Survived")
print(fareNotSurvivedSr.describe())
print("All together")
print(m.analyzedDF.Fare.describe())
#So there are differences

#Let's try to guess whtat's happening with the 0s in the Fare
m.analyzedDF[m.analyzedDF.Fare==0].Survived

#To the one who survived we'll set the Fare of the ones who survived. The opposite which those who didn't
meanFareSurvived=fareSurvivedSr.mean()
meanFareNotSurvived=fareNotSurvivedSr.mean()

m.analyzedDF.loc[lambda x: ((x.Fare==0) & (x.Survived==0)), 'Fare']=meanFareNotSurvived
m.analyzedDF.loc[lambda x: ((x.Fare==0) & (x.Survived==1)), 'Fare']=meanFareSurvived

#We'll see the differences after our changes
print("--------------------")
print("After changes")
print("--------------------")

fareNotSurvivedSr=m.analyzedDF[m.analyzedDF.Survived==0].Fare
fareSurvivedSr=m.analyzedDF[m.analyzedDF.Survived==1].Fare
print("Fare Survived")
print(fareSurvivedSr.describe())
print ("Fare not Survived")
print(fareNotSurvivedSr.describe())
print("All together")
print(m.analyzedDF.Fare.describe())

#%% Comparison of sex and class with Survived

m.analyzedDF.loc[lambda x: x.Sex=='male'].Survived.describe()
m.analyzedDF.loc[lambda x: x.Sex=='female'].Survived.describe()
print ("Sex matters!!!")

print(m.analyzedDF.groupby([m.analyzedDF.Pclass, m.analyzedDF.Survived])['PassengerId'].count())
print ("Class obviously matters!!!")

#%% Correlation of Age with Surviving percentage.
# We are going to make a cut on Age and group to view the percentage of Surviving

ageCut = pd.cut(m.analyzedDF.Age,6)

print (m.analyzedDF.groupby([ageCut, m.analyzedDF['Survived']])['PassengerId'].count() / m.analyzedDF.groupby(ageCut)['PassengerId'].count() *100)
print ("Age also is a difference in surviving options")

#%% We also do the same study with Parch and Sibsp
print(m.analyzedDF.groupby(['Parch', 'Survived'])['PassengerId'].count() / m.analyzedDF.groupby('Parch')['PassengerId'].count() *100)
print(m.analyzedDF.groupby(['SibSp', 'Survived'])['PassengerId'].count() / m.analyzedDF.groupby('SibSp')['PassengerId'].count() *100)
print("We consider Parch and SibSp as important fields too")

#%% Nulls on data (train and test)
print (m.analyzedDF.isnull().any())
print (m.dataTestDF.isnull().any())
