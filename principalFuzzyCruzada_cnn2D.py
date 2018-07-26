from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import metodosPrincipais as mp
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

diretorio_treinamento = './FuzzyBase/train/'
diretorio_validacao = './FuzzyBase/validation/'

X_train, X_val, y_train, y_val = mp.obterConjuntoTreinamentoValidacao(diretorio_treinamento,diretorio_validacao)

previsores = X_train.reshape(X_train.shape[0], 5, 12, 1)
previsores = previsores.astype('float32')
previsores /= 255
#classe = to_categorical(y, 10)
#classe = classe.astype('float32')

kfold = StratifiedKFold(n_splits=5, shuffle=True)
resultados = []

a = np.zeros(5)
b = np.zeros(shape=(classe.shape[0], 1))

for indice_treinamento, indice_teste in kfold.split(previsores, b):
    #print('Indices treinamento', indice_treinamento, 'Indice teste', indice_teste)
    classificador = Sequential()
    classificador.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2, 2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax'))
    classificador.compile(loss='categorical_crossentropy', optimizer='adam', 
                          metrics=['accuracy'])
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento], 
                      batch_size=128, epochs=5)
    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])
    
media = sum(resultados) / len(resultados)