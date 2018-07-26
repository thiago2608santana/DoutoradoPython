#import numpy as np
import scipy.io as sp
#import matplotlib.pyplot as plt
#import os
#import io
import metodosPrincipais as mp

#Definir um diretório base para armazenamento futuro dos dados de treinamento, validação e teste
base_dir = './DiretorioBase'

#Carregar os dados dos HD-sEMG
vasto_controle_10_1 = sp.loadmat('vasto_controle_max_vl1.mat')
vasto_diabeticos_10_1 = sp.loadmat('vasto_diabeticos_max_vl1.mat')
biceps_controle_10_1 = sp.loadmat('biceps_controle_max_bf1.mat')
biceps_diabeticos_10_1 = sp.loadmat('biceps_diabeticos_max_bf1.mat')
gastrocnemio_controle_10_1 = sp.loadmat('gastrocnemio_controle_max_gm1.mat')
gastrocnemio_diabeticos_10_1 = sp.loadmat('gastrocnemio_diabeticos_max_gm1.mat')
tibial_controle_10_1 = sp.loadmat('tibial_controle_max_ta1.mat')
tibial_diabeticos_10_1 = sp.loadmat('tibial_diabeticos_max_ta1.mat')

#Definir a quantidade de imagens do EMG de cada pessoa e o nome do diretório a ser criado para salvar essas imagens
qtdImagens = 500
diretorio_imagens_controle = 'ImagensGrupoControle'
diretorio_imagens_diabetico = 'ImagensGrupoDiabeticos'

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

mp.criarDiretorio(diretorio_imagens_controle)
mp.criarDiretorio(diretorio_imagens_diabetico)
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, vasto_controle_10_1, diretorio_imagens_controle, 'vl')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, vasto_diabeticos_10_1, diretorio_imagens_diabetico, 'vl')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, biceps_controle_10_1, diretorio_imagens_controle, 'bf')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, biceps_diabeticos_10_1, diretorio_imagens_diabetico, 'bf')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, gastrocnemio_controle_10_1, diretorio_imagens_controle, 'gm')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, gastrocnemio_diabeticos_10_1, diretorio_imagens_diabetico, 'gm')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, tibial_controle_10_1, diretorio_imagens_controle, 'ta')
mp.salvarImagensEmDisco(qtdAmostras, qtdImagens, tibial_diabeticos_10_1, diretorio_imagens_diabetico, 'ta')

mp.prepararDados(base_dir,f'./{diretorio_imagens_controle}',f'./{diretorio_imagens_diabetico}',
                 qtdTrain,qtdValidation,qtdTest)