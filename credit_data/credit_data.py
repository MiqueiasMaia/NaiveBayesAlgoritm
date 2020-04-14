# -*- coding: utf-8 -*-

import pandas
import numpy
import matplotlib.pyplot as plotter
base = pandas.read_csv('Naive-Bayes/credit_data/credit_data.csv')

# base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean() #não influenciou no resultado

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# from sklearn.preprocessing import StandardScaler #não influenciou no resultado
# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)


# leituras = pandas.read_csv('Naive-Bayes/credit_data/reads.csv')
# leituras = leituras.iloc[:,:].values
# plotter.plot(previsoes)
# plotter.show()
