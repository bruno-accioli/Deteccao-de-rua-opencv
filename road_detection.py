import numpy as np
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from math import floor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import ceil
import time

def mudarEscala(img, max_valor=255.0):
    escala = np.max(img)/max_valor
    return (img/escala).astype(np.uint8)

def aplicarThreshold(img, threshold_min = 0, threshold_max = 255):
    img_saida = np.zeros_like(img)
    img_saida[(img >= threshold_min) & (img <= threshold_max)] = 1
    return img_saida

def AplicarSobel(img, k = 5):
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k))
    mag = np.sqrt(sobelx**2 + sobely**2)
    return mudarEscala(mag)
        

def playVideo(video):
    while(video.isOpened()):
        ret, frame = video.read()  
        cv2.imshow('frame',frame)
        if  cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

def mostrarImagem(imagem, mascara = None):
    if mascara:
        _, _, ch = imagem.shape
        mask = list()
        for i in range(ch):
            mask.append(mascara)
        mascara = np.array(mask)
        imagem = np.multiply(imagem, mascara)
    cv2.imshow("Imagem", imagem)
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def salvarImagem(imagem, numero):
    cv2.imwrite('imagens/teste'+str(numero)+'.png', imagem)
    
def pegarFrame(cap, numeroDoFrame):
    # numero total de frames do video
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # checar valor valido de frames
    if numeroDoFrame >= 0 and numeroDoFrame <= totalFrames:
        # colocar o video na posicao certa do frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, numeroDoFrame)
    else:
        raise ValueError('O número do frame não pode ser negativo e nem maior que o numero total')    
    
    ret, frame = cap.read()
    mostrarImagem(frame)
    
    return frame

# Função que diz o numero do frame dado o tempo exato do video
def CalcularNumeroDoFrame(video, horas = 0, mins = 0, segs = 0, ms = 0):
    tempoTotalEmSegs = horas * 3600 + mins * 60 + segs + ms * 0.001
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    numeroDoFrame = floor(tempoTotalEmSegs * FPS)
    
    if numeroDoFrame >= 0 and numeroDoFrame <= totalFrames:
        return numeroDoFrame
    else:
        print('O tempo do frame (%2d:%2d:%2d:%2d) é maior que o tempo do video.'
              %(horas, mins, segs,ms))
        return None
        
def processarImagem(img, pontosRua, sobelSize = 5, threshold_min = 50):

    # Planificar perspectiva da rua
    destino = np.float32([[0,0], [LARGURA, 0], [LARGURA, ALTURA], [0, ALTURA]])
    MatDist = cv2.getPerspectiveTransform(pontosRua, destino)
    
    # Monstrar pontos
    img2 = img.copy()
    img2 = cv2.circle(img2, tuple(topoEsq), 3, (255,0,0), thickness=1, lineType=8, shift=0) 
    img2 = cv2.circle(img2, tuple(topoDir), 3, (0,255,0), thickness=1, lineType=8, shift=0)
    img2 = cv2.circle(img2, tuple(baseDir), 3, (0,0,255), thickness=1, lineType=8, shift=0)
    img2 = cv2.circle(img2, tuple(baseEsq), 3, (0,255,255), thickness=1, lineType=8, shift=0)
    
    mostrarImagem(img2)
    
    img_distorcida = cv2.warpPerspective(img, MatDist, (LARGURA, ALTURA))
    mostrarImagem(img_distorcida)
    
    # Coverter para HSV
    img_hsv = cv2.cvtColor(img_distorcida, cv2.COLOR_BGR2HSV)
    
    #Converter to grayscale
    img_bw = cv2.cvtColor(img_distorcida, cv2.COLOR_BGR2GRAY)
    
    # Criar máscara de faixa amarela
    AmareloHsvMin  = np.array([ 0, 80, 200])
    AmareloHsvMax = np.array([ 40, 255, 255])
    mascaraAmarelo = cv2.inRange(img_hsv, AmareloHsvMin, AmareloHsvMax)
    
    # Criar máscara de faixa branca
    BrancoHsvMin  = np.array([50, 13, 160])
    BrancoHsvMax = np.array([255, 180, 255])
    mascaraBranco = cv2.inRange(img_hsv, BrancoHsvMin, BrancoHsvMax)
    mostrarImagem(mascaraBranco)
    
    
    # Aplicar Sobel    
    sobel = AplicarSobel(img_bw, k = 5)
    sobel = aplicarThreshold(sobel, threshold_min = threshold_min, threshold_max = 255)
    mostrarImagem(sobel)
    
    # Mascaras Combinadas
    mascara = cv2.bitwise_or(mascaraAmarelo, mascaraBranco)
    mascara = cv2.bitwise_or(mascara, sobel)
#    mascara = cv2.bitwise_or(mascara, sobely)
#    mostrarImagem(mascara)
    return mascara
#    res = cv2.bitwise_and(img_distorcida,img_distorcida, mask=mascara)
#    mostrarImagem(res)

def pegarPontosDaRua(larguraBaseRazao, larguraTopoRazao, alturaRazao, 
                     pixelsDoCarroRazao, DeslocamentoLateralRazao = 0):
    
    pontoTopoEsquerda = [round(LARGURA * (1 - larguraTopoRazao + DeslocamentoLateralRazao) / 2),
                         round(ALTURA * (1 - alturaRazao - pixelsDoCarroRazao))]
    pontoTopoDireita = [round(pontoTopoEsquerda[0] + (LARGURA * larguraTopoRazao)),
                        pontoTopoEsquerda[1]]
    pontoBaseEsquerda = [round(LARGURA * (1 - larguraBaseRazao + DeslocamentoLateralRazao) / 2),
                         pontoTopoEsquerda[1] + round(ALTURA * alturaRazao)]
    pontoBaseDireita = [round(pontoBaseEsquerda[0] + (LARGURA * larguraBaseRazao)),
                        pontoBaseEsquerda[1]]
    
    return pontoTopoEsquerda, pontoTopoDireita, pontoBaseDireita, pontoBaseEsquerda

def gerarCurvasDaFaixa(img, n_janelas, margem=50, tolerancia = 25):
    # Achar picos do histograma da metade de baixo da imagem    
    minpix = 50
    histograma = np.sum(img[int(ALTURA/2):,:], axis=0)
    metade = int(LARGURA/2)
    picoEsquerdoBase = np.argmax(histograma[:metade])
    picoDireitoBase = np.argmax(histograma[metade:]) + metade
    print(np.argmax(histograma[metade:]))
    
    
    mascara = np.zeros_like(img)
    coordenadas = img.nonzero()
    coordenadasy = np.array(coordenadas[0])
    coordenadasx = np.array(coordenadas[1])
    
    faixaEsquerda = np.array([])
    faixaDireita = np.array([])
    
    # Para cada janela horizontal, gerar uma mascara
    ultimoPicoEsquerdo = picoEsquerdoBase
    ultimoPicoDireito = picoDireitoBase
    picoEsquerdo = picoEsquerdoBase
    picoDireito = picoDireitoBase
    alturaJanela = ceil(ALTURA/n_janelas)
    cont=1
    for inicioJanela_Y in np.arange(0, ALTURA, alturaJanela):
        fimJanela_Y = inicioJanela_Y + alturaJanela
#        histograma = np.sum(img[inicioJanela_Y:fimJanela_Y,:], axis=0)
#        picoEsquerdo = np.argmax(histograma[:metade])
#        picoDireito = np.argmax(histograma[metade:]) + metade    
        print('Iteraçao %d' %cont)
        print('Pico Esquerdo: %d' %picoEsquerdo)
        print('Pico Direito: %d\n' %picoDireito)
        cont += 1
        if (not checarPicoValido(picoEsquerdo, ultimoPicoEsquerdo, tolerancia)):
            picoEsquerdo = ultimoPicoEsquerdo
        
        inicioJanelaEsquerda_X = picoEsquerdo-margem
        fimJanelaEsquerda_X = picoEsquerdo+margem
        mascara[inicioJanela_Y:fimJanela_Y, inicioJanelaEsquerda_X:fimJanelaEsquerda_X] = 1

        if (not checarPicoValido(picoDireito, ultimoPicoDireito, tolerancia)):
            picoDireito = ultimoPicoDireito
        inicioJanelaDireita_X = picoDireito-margem
        fimJanelaDireita_X = picoDireito+margem
        mascara[inicioJanela_Y:fimJanela_Y, inicioJanelaDireita_X:fimJanelaDireita_X] = 1
        
        faixaEsquerdaIds = ((coordenadasx >= inicioJanelaEsquerda_X) & (coordenadasx <= fimJanelaEsquerda_X) &
                            (coordenadasy >= inicioJanela_Y) & (coordenadasy >= fimJanela_Y)).nonzero()[0]

        faixaDireitaIds = ((coordenadasx >= inicioJanelaDireita_X) & (coordenadasx <= fimJanelaDireita_X) &
                            (coordenadasy >= inicioJanela_Y) & (coordenadasy >= fimJanela_Y)).nonzero()[0]
        
        faixaDireita = np.append(faixaDireita, faixaDireitaIds, axis=0)
        faixaEsquerda = np.append(faixaEsquerda, faixaEsquerdaIds, axis=0)
        
        ultimoPicoEsquerdo = picoEsquerdo
        ultimoPicoDireito = picoDireito
        
        if len(faixaEsquerdaIds) >= minpix:
            picoEsquerdo = np.int(np.mean(coordenadasx[faixaEsquerdaIds]))
            
        if len(faixaDireitaIds) >= minpix:
            picoDireito = np.int(np.mean(coordenadasx[faixaDireitaIds]))
    
    faixaEsquerdax = coordenadasx[faixaEsquerda.astype(int)]
    faixaEsquerday = coordenadasy[faixaEsquerda.astype(int)] 
    faixaDireitax = coordenadasx[faixaDireita.astype(int)]
    faixaDireitay = coordenadasy[faixaDireita.astype(int)] 
    
    # Fit a second order polynomial to each
    faixaEsquerda_fit = np.polyfit(faixaEsquerday, faixaEsquerdax, 2)
    faixaDireita_fit = np.polyfit(faixaDireitay, faixaDireitax, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    faixaEsquerda_fitx = faixaEsquerda_fit[0]*ploty**2 + faixaEsquerda_fit[1]*ploty + faixaEsquerda_fit[2]
    faixaDireita_fitx = faixaDireita_fit[0]*ploty**2 + faixaDireita_fit[1]*ploty + faixaDireita_fit[2]
    
    # Color the left lane red and the right lane blue
    out_img = np.uint8(np.dstack((img, img, img))*255)
    out_img[coordenadasy[faixaEsquerdaIds], coordenadasx[faixaEsquerdaIds]] = (255, 0, 0)
    out_img[coordenadasy[faixaDireitaIds], coordenadasx[faixaDireitaIds]] = (0, 0, 255)

    return mascara, out_img, faixaEsquerda_fitx, faixaEsquerda_fit, faixaDireita_fitx, faixaDireita_fit

def checarPicoValido(pico, ultimoPico, tol):
    if (abs(pico-ultimoPico) >= tol):
        return False

    if (pico == 0):
        return False

    return True

# cortar video
t1 = 28 * 60 #segundos
t2 = 32 * 60 #segundos
#ffmpeg_extract_subclip("video_inteiro_720.mp4", t1, t2, targetname="video_parte_720.mp4")

# abrir video
video = cv2.VideoCapture('video_parte_720.mp4')

# Informaçoes do video
LARGURA = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
ALTURA = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = video.get(cv2.CAP_PROP_FPS)


# Gerar imagens de teste e salvar
'''
temposImagensDeTeste = [[0, 5], [0, 13], [0, 34], [1, 12], [1, 25], [2, 54], [3, 23]]
imagensTeste = list()
for i in range(len(temposImagensDeTeste)):    
    mins = temposImagensDeTeste[i][0]
    segs = temposImagensDeTeste[i][1]
    numeroDoFrame = CalcularNumeroDoFrame(video, mins = mins,  segs = segs)
    imagem = pegarFrame(video, numeroDoFrame)
    imagensTeste.append(imagem)
    salvarImagem(imagem, i)
'''

# Carregar imagens de teste
imagensTeste = list()
cont = 0
while True:
    temp = cv2.imread('imagens/teste'+str(cont)+'.png')
    if temp is None:
        break
    else:
        imagensTeste.append(temp)
        cont += 1
    
    
# Pontos que delimitam a rua
topoEsq, topoDir, baseDir, baseEsq = pegarPontosDaRua(0.65, 0.135, 0.20, 0.12, -0.005)
pontosRua = np.float32([topoEsq, topoDir, baseDir, baseEsq])
#processarImagem(imagensTeste[0], pontosRua)
#mostrarImagem(imagensTeste[0])

# Sobel
#for i in range(len(imagensTeste)):
#    mostrarImagem(imagensTeste[i])
#    processarImagem(imagensTeste[i], pontosRua, 50)
#mostrarImagem(imagensTeste[0])
mascara = processarImagem(imagensTeste[0], pontosRua, 50)
mostrarImagem(mascara)
cv2.imwrite('mascara.png', mascara)
start = time.time()
mascara2, out_img, esqx, _, dirx, _ = gerarCurvasDaFaixa(mascara, 7)
end=time.time()
t=end-start
print(t)
mostrarImagem(mascara2 * 255)
mostrarImagem(out_img)

t = np.zeros((720, 1280))
#ids = (cx >= 640).nonzero()[0]
for i in range(len(cx)):
    if(cy[i] >= 640):
        t[cx[i], cy[i]] = 255

mostrarImagem(t)

y = np.linspace(0, mascara.shape[0]-1, mascara.shape[0] )
y = y.astype(int)
t = np.zeros((720, 1280))
esqx = esqx.astype(int)
dirx = dirx.astype(int)
for i in range(len(y)):
    t[y[i], esqx[i]] = 255
    t[y[i], dirx[i]] = 255

mostrarImagem(t)

#hist = np.histogram(mascara)

#km = KMeans(n_clusters = 2, max_iter=3)
#start = time.time()
#hist = mascara.sum(axis=0)
##plt.plot(hist)
#teste = np.flatnonzero(mascara)%LARGURA
#plt.hist(a)

#km.fit(hist.reshape(-1,1))
#km.fit(teste.reshape(-1,1))
#end=time.time()
#t=end-start
#print(km.cluster_centers_)
#
#start = time.time()
#end=time.time()
#t=end-start
#print(t)
    


#BGR
#img = np.copy(imagensTeste[0])
#img = cv2.circle(img, tuple(topoEsq), 3, (255,0,0), thickness=1, lineType=8, shift=0) 
#img = cv2.circle(img, tuple(topoDir), 3, (0,255,0), thickness=1, lineType=8, shift=0)
#img = cv2.circle(img, tuple(baseDir), 3, (0,0,255), thickness=1, lineType=8, shift=0)
#img = cv2.circle(img, tuple(baseEsq), 3, (0,255,255), thickness=1, lineType=8, shift=0)
#
#mostrarImagem(img)



