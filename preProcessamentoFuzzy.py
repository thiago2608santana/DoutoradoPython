import scipy.io as sp
#import os
#import io
import metodosPrincipais as mp

#Definir um diretório base para armazenamento futuro dos dados de treinamento, validação e teste
base_dir = './FuzzyBase'

#Carregar os dados dos HD-sEMG
vasto_controle_10_1 = sp.loadmat('fuzzy_biceps_controle_10.mat')
vasto_diabeticos_10_1 = sp.loadmat('fuzzy_biceps_diabeticos_10.mat')
biceps_controle_10_1 = sp.loadmat('fuzzy_gastrocnemio_controle_10.mat')
biceps_diabeticos_10_1 = sp.loadmat('fuzzy_gastrocnemio_diabeticos_10.mat')
gastrocnemio_controle_10_1 = sp.loadmat('fuzzy_tibial_controle_10.mat')
gastrocnemio_diabeticos_10_1 = sp.loadmat('fuzzy_tibial_diabeticos_10.mat')
tibial_controle_10_1 = sp.loadmat('fuzzy_vasto_controle_10.mat')
tibial_diabeticos_10_1 = sp.loadmat('fuzzy_vasto_diabeticos_10.mat')

#Definir a quantidade de imagens do EMG de cada pessoa e o nome do diretório a ser criado para salvar essas imagens
qtdImagens = 1
diretorio_fuzzy_controle = 'ImagensFuzzyControle'
diretorio_fuzzy_diabetico = 'ImagensFuzzyDiabeticos'

#Definir a quantidade de voluntários diferentes em cada grupo
qtdAmostras = 10

#Definir a quantidade de imagens nos conjuntos de treinamento, validação e teste
qtdPercentTrain = 50
qtdPercentValidation = 30
qtdPercentTest = 20
qtdTrain, qtdValidation, qtdTest = mp.definirTreinamentoValidacao(qtdPercentTrain, 
                                                                  qtdPercentValidation, 
                                                                  qtdPercentTest, 
                                                                  qtdAmostras, qtdImagens)

mp.criarDiretorio(diretorio_fuzzy_controle)
mp.criarDiretorio(diretorio_fuzzy_diabetico)

mp.salvarImagensFuzzy(qtdAmostras, vasto_controle_10_1, diretorio_fuzzy_controle, 'vl', 'controle')
mp.salvarImagensFuzzy(qtdAmostras, vasto_diabeticos_10_1, diretorio_fuzzy_diabetico, 'vl', 'diabetico')
mp.salvarImagensFuzzy(qtdAmostras, biceps_controle_10_1, diretorio_fuzzy_controle, 'bf', 'controle')
mp.salvarImagensFuzzy(qtdAmostras, biceps_diabeticos_10_1, diretorio_fuzzy_diabetico, 'bf', 'diabetico')
mp.salvarImagensFuzzy(qtdAmostras, gastrocnemio_controle_10_1, diretorio_fuzzy_controle, 'gm', 'controle')
mp.salvarImagensFuzzy(qtdAmostras, gastrocnemio_diabeticos_10_1, diretorio_fuzzy_diabetico, 'gm', 'diabetico')
mp.salvarImagensFuzzy(qtdAmostras, tibial_controle_10_1, diretorio_fuzzy_controle, 'ta', 'controle')
mp.salvarImagensFuzzy(qtdAmostras, tibial_diabeticos_10_1, diretorio_fuzzy_diabetico, 'ta', 'diabetico')

mp.prepararDadosFuzzy(base_dir,f'./{diretorio_fuzzy_controle}',f'./{diretorio_fuzzy_diabetico}',
                 qtdTrain,qtdValidation,qtdTest)