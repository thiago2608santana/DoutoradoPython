import os, shutil
import numpy as np
#import matplotlib.pyplot as plt
import scipy.misc

#Link interessante sobre array como imagem
#https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image

#Função que cria um diretório para salvar as imagens caso não exista
def criarDiretorio(nomeDiretorio):
    if not os.path.isdir(f'./{nomeDiretorio}'):
        os.mkdir(f'./{nomeDiretorio}')

#Função para extrair imagens(frames) separadas(os) de um único voluntário
def obterPessoas(dados, indice):
    matriz = []
    for i in range(5):
        linha = []
        for j in range(12):
            linha.append(dados[i][j][indice])
        matriz.append(linha)
    return matriz

#Função auxiliar que é usada como parâmetro da função sorted() para ordenar a lista de figuras em uma pasta
def takeFirstAndLast(element):
    element_splited = []
    element_splited.append(int(element.split('_')[0]))
    element_splited.append(int(element.split('_')[4].split('.')[0]))
    return element_splited
    
#Salvar em disco as imagens convertidas para escala de cinza de acordo com os parâmetros previamente definidos
def salvarImagensEmDisco(qtdAmostras, qtdImagens, dados, diretorio, nomeDados):
    for i in range(qtdAmostras):
        for j in range(qtdImagens):
            matriz = obterPessoas(dados['dados_prontos'][i][0],j)
            dadoVoluntarioControle = np.asarray(matriz)
            dadoVoluntarioControle[4][0] = 0
            maximo = np.amax(dadoVoluntarioControle)#
            minimo = np.amin(dadoVoluntarioControle)#
            imagem = scipy.misc.toimage(dadoVoluntarioControle, cmin=minimo, cmax=maximo)#
            #plt.gray()
            #plt.imshow(dadoVoluntarioControle)
            if nomeDados == 'bf':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_biceps_{j}.jpg',dadoVoluntarioControle)
                imagem.save(f'./{diretorio}/{i}_imagem_controle_biceps_{j}.jpg')
            elif nomeDados == 'gm':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_gastrocnemio_{j}.jpg',dadoVoluntarioControle)
                imagem.save(f'./{diretorio}/{i}_imagem_controle_gastrocnemio_{j}.jpg')
            elif nomeDados == 'ta':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_tibial_{j}.jpg',dadoVoluntarioControle)
                imagem.save(f'./{diretorio}/{i}_imagem_controle_tibial_{j}.jpg')
            else:
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_vasto_{j}.jpg',dadoVoluntarioControle)
                imagem.save(f'./{diretorio}/{i}_imagem_controle_vasto_{j}.jpg')

#Função que define a quantidade de amostras nos grupos de treino, validação e teste
def definirTreinamentoValidacao(qtdPercentTrain, qtdPercentValidation, qtdPercentTest, qtdAmostras, qtdImagens):
    qtdAmostras = 2*qtdAmostras
    train = round((qtdPercentTrain*qtdAmostras)/100)
    validation = round((qtdPercentValidation*qtdAmostras)/100)
    test = round((qtdPercentTest*qtdAmostras)/100)
    
    qtdTrain = 2*train*qtdImagens
    qtdValidation = 2*validation*qtdImagens
    qtdTest = 2*test*qtdImagens
    
    return qtdTrain, qtdValidation, qtdTest

#Função para converter uma imagem colorida em escala de cinza
def rgb2gray(rgb):
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray
    
#Função que prepara os dados, criando diretórios caso não existam e copiando arquivos nos devidos lugares
def prepararDados(base_dir, controle_dir, diabetico_dir, qtdTrain, qtdValidation, qtdTest):
    itens_controle_dir = sorted(os.listdir(controle_dir), key=takeFirstAndLast)
    itens_diabetico_dir = sorted(os.listdir(diabetico_dir), key=takeFirstAndLast)
    
    criarDiretorio(base_dir)
    
    #Criar diretórios principais de treinamento, validação e teste
    train_dir = os.path.join(base_dir,'train')
    criarDiretorio(train_dir)
    validation_dir = os.path.join(base_dir,'validation')
    criarDiretorio(validation_dir)
    test_dir = os.path.join(base_dir,'test')
    criarDiretorio(test_dir)
    
    #Criar subdiretórios de treinamento, validação e teste para controle e diabético
    train_controle_dir = os.path.join(train_dir,'controle')
    criarDiretorio(train_controle_dir)
    train_diabetico_dir = os.path.join(train_dir,'diabetico')
    criarDiretorio(train_diabetico_dir)
    
    validation_controle_dir = os.path.join(validation_dir,'controle')
    criarDiretorio(validation_controle_dir)
    validation_diabetico_dir = os.path.join(validation_dir,'diabetico')
    criarDiretorio(validation_diabetico_dir)
    
    test_controle_dir = os.path.join(test_dir,'controle')
    criarDiretorio(test_controle_dir)
    test_diabetico_dir = os.path.join(test_dir,'diabetico')
    criarDiretorio(test_diabetico_dir)
    
    #Copiar imagens dos diretórios base para subdiretórios de treinamento, validação e teste
    for i in range(qtdTrain):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(train_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(validation_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(test_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(train_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(validation_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)

    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(test_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)