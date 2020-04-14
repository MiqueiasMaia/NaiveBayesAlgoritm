# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Naive-Bayes/censo/censo.csv')
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEnconderPrevisores = LabelEncoder()
previsores[:,1] = labelEnconderPrevisores.fit_transform(previsores[:,1])
previsores[:,3] = labelEnconderPrevisores.fit_transform(previsores[:,3])
previsores[:,5] = labelEnconderPrevisores.fit_transform(previsores[:,5])
previsores[:,6] = labelEnconderPrevisores.fit_transform(previsores[:,6])
previsores[:,7] = labelEnconderPrevisores.fit_transform(previsores[:,7])
previsores[:,8] = labelEnconderPrevisores.fit_transform(previsores[:,8])
previsores[:,9] = labelEnconderPrevisores.fit_transform(previsores[:,9])
previsores[:,13] = labelEnconderPrevisores.fit_transform(previsores[:,13])

### influencia negativamente no acerto
# oneHotEnconder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
# previsores = oneHotEnconder.fit_transform(previsores).toarray()

# labelEnconderClasse = LabelEncoder()
# classe = labelEnconderClasse.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

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
