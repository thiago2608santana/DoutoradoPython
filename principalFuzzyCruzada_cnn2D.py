#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import metodosPrincipais as mp
#from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

diretorio_treinamento = './FuzzyBase/train/'
diretorio_validacao = './FuzzyBase/validation/'

X_train, X_val, y_train, y_val = mp.obterConjuntoTreinamentoValidacao(diretorio_treinamento,diretorio_validacao)

previsores = X_train.reshape(X_train.shape[0], 5, 12, 1)
previsores = previsores.astype('float32')
previsores /= 255

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

caminho_controle = './ImagensFuzzyControle/'
caminho_diabeticos = './ImagensFuzzyDiabeticos/'
k_fold = 10
qtdAmostrasPorGrupo = 8
qtdTrain = 16
qtdValidation = 2
qtdCondicoes = 4
qtdRepeticoes = 2
qtdMusculos = 4
indices_treinamento, indices_teste = mp.validacaoCruzada(caminho_controle,
                                                         caminho_diabeticos,
                                                         k_fold,
                                                         qtdAmostrasPorGrupo,
                                                         qtdTrain,qtdValidation,
                                                         qtdCondicoes,
                                                         qtdRepeticoes,
                                                         qtdMusculos)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
#rkfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

vetor_indice = np.zeros(shape=(y_train.shape[0], 1))

resultados = []

for indice_treinamento, indice_teste in kfold.split(previsores, vetor_indice):
    #print('Indices treinamento', indice_treinamento, 'Indice teste', indice_teste)
    classificador = Sequential()
    classificador.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(5, 12, 1), activation='relu', padding='same'))
    classificador.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classificador.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    classificador.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    classificador.add(Flatten())
    classificador.add(Dense(units=512, activation='relu'))
    classificador.add(Dense(units=256, activation='relu'))
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(loss='binary_crossentropy', optimizer='adam', 
                          metrics=['accuracy'])
    classificador.fit(previsores[indice_treinamento], y_train[indice_treinamento], 
                      batch_size=10, epochs=20)
    precisao = classificador.evaluate(previsores[indice_teste], y_train[indice_teste])
    resultados.append(precisao[1])
    
media = sum(resultados) / len(resultados)