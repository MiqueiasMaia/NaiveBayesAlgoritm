# -*- coding: utf-8 -*-

import pandas

base = pandas.read_csv('Naive-Bayes/credit_risk/risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
previsores[:,0] = labelEncoder.fit_transform(previsores[:,0])
previsores[:,1] = labelEncoder.fit_transform(previsores[:,1])
previsores[:,2] = labelEncoder.fit_transform(previsores[:,2])
previsores[:,3] = labelEncoder.fit_transform(previsores[:,3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe) #treinamento - gera tabela

resultado = classificador.predict([[0,0,1,2],[3,0,0,0]]) #calculos

print(classificador.class_prior_)