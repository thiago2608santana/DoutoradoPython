import os, shutil
import numpy as np
#import scipy.misc
import PIL.Image as pil

#Link interessante sobre array como imagem
#https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
#Amostra 0 do bícpes dos diabéticos tem valores muito extremos no frame 4118
#Amostra 1 do bíceps dos controle tem valores extremos no frame 8832

#Obter valor máximo e mínimo de cada trial
def obterMinMax(dados):
    minimo = 0
    maximo = 0
    
    for i in range(len(dados[0][0][:])):
        frame = obterFrames(dados,i)
        matriz = np.asarray(frame)
        matriz[4][0] = 0
        if np.amax(matriz) > maximo:
            maximo = np.amax(matriz)
        if np.amin(matriz) < minimo:
            minimo = np.amin(matriz)
            
    return minimo, maximo

#Obter valor mínimo e máximo de cada músculo, no intuito de normalizar os dados
#do grupo diabéticos com base nos dados do grupo controle
def obterMinMaxPorMusculo(qtdAmostras, dados):
    minimo = 0
    maximo = 0
    
    for i in range(qtdAmostras):
        for j in range(len(dados['dados_prontos'][i][0][0][0][:])):
            frame = obterFrames(dados['dados_prontos'][i][0],j)
            matriz = np.asarray(frame)
            matriz[4][0] = 0
            if np.amax(matriz) > maximo:
                maximo = np.amax(matriz)
                num_amostra_max = i
                num_frame_max = j
            if np.amin(matriz) < minimo:
                minimo = np.amin(matriz)
                num_amostra_min = i
                num_frame_min = j
    print(f'Amostra do máximo: {num_amostra_max}')
    print(f'Amostra do mínimo: {num_amostra_min}')
    print(f'Frame do máximo: {num_frame_max}')
    print(f'Frame do mínimo: {num_frame_min}')
    return minimo, maximo

#Função que cria um diretório para salvar as imagens caso não exista
def criarDiretorio(nomeDiretorio):
    if not os.path.isdir(f'./{nomeDiretorio}'):
        os.mkdir(f'./{nomeDiretorio}')

#Função para extrair imagens(frames) separadas(os) de um único voluntário
def obterFrames(dados, indice):
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
def salvarImagensEmDisco(qtdAmostras, qtdImagens, dados, diretorio, minimo, maximo, nomeDados, grupoDados):
    for i in range(qtdAmostras):
        for j in range(qtdImagens):
            frame = obterFrames(dados['dados_prontos'][i][0],j)
            matriz = np.asarray(frame)
            matriz[4][0] = 0
            imagem = pil.fromarray(matriz, mode='L')
            #imagem = scipy.misc.toimage(matriz, cmin=minimo, cmax=maximo)
            #plt.gray()
            #plt.imshow(dadoVoluntarioControle)
            if nomeDados == 'bf':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_biceps_{j}.jpg',dadoVoluntarioControle)
                if grupoDados == 'controle':
                    imagem.save(f'./{diretorio}/{i}_imagem_controle_biceps_{j}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_imagem_diabetico_biceps_{j}.jpg')
            elif nomeDados == 'gm':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_gastrocnemio_{j}.jpg',dadoVoluntarioControle)
                if grupoDados == 'controle':
                    imagem.save(f'./{diretorio}/{i}_imagem_controle_gastrocnemio_{j}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_imagem_diabetico_gastrocnemio_{j}.jpg')
            elif nomeDados == 'ta':
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_tibial_{j}.jpg',dadoVoluntarioControle)
                if grupoDados == 'controle':
                    imagem.save(f'./{diretorio}/{i}_imagem_controle_tibial_{j}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_imagem_diabetico_tibial_{j}.jpg')
            else:
                #plt.imsave(f'./{diretorio}/{i}_imagem_controle_vasto_{j}.jpg',dadoVoluntarioControle)
                if grupoDados == 'controle':
                    imagem.save(f'./{diretorio}/{i}_imagem_controle_vasto_{j}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_imagem_diabetico_vasto_{j}.jpg')

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

#------------------------------------------------------------------------------
#Métodos para tratamento dos dados processados utilizando lógica fuzzy
#------------------------------------------------------------------------------
#Rearranjar os dados para o formato de matriz
def organizarDadosFuzzy(dados, indice):
    vetorFuzzy = dados['FuzEn'][indice][0][0]
    matrizFuzzy = np.reshape(vetorFuzzy, (5, 12))
    return matrizFuzzy

def salvarImagensFuzzy(qtdAmostras, dados, diretorio, nomeDados, grupoDados):
    for i in range(qtdAmostras):
        matriz = organizarDadosFuzzy(dados,i)
        imagem = pil.fromarray(matriz, mode='L')
        
        if nomeDados == 'bf':
            if grupoDados == 'controle':
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_biceps.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_biceps.jpg')
        elif nomeDados == 'gm':
            if grupoDados == 'controle':
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_gastrocnemio.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_gastrocnemio.jpg')
        elif nomeDados == 'ta':
            if grupoDados == 'controle':
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_tibial.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_tibial.jpg')
        else:
            if grupoDados == 'controle':
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_vasto.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_imagem_fuzzy_vasto.jpg')
                
def prepararDadosFuzzy(base_dir, controle_dir, diabetico_dir, qtdTrain, qtdValidation, qtdTest):
    itens_controle_dir = os.listdir(controle_dir)
    itens_diabetico_dir = os.listdir(diabetico_dir)
    
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