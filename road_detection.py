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
#    img2 = img.copy()
#    img2 = cv2.circle(img2, tuple(topoEsq), 3, (255,0,0), thickness=1, lineType=8, shift=0) 
#    img2 = cv2.circle(img2, tuple(topoDir), 3, (0,255,0), thickness=1, lineType=8, shift=0)
#    img2 = cv2.circle(img2, tuple(baseDir), 3, (0,0,255), thickness=1, lineType=8, shift=0)
#    img2 = cv2.circle(img2, tuple(baseEsq), 3, (0,255,255), thickness=1, lineType=8, shift=0)
#    
#    mostrarImagem(img2)
    
    img_distorcida = cv2.warpPerspective(img, MatDist, (LARGURA, ALTURA))
#    mostrarImagem(img_distorcida)
    
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
#    mostrarImagem(mascaraBranco)    
    
    # Aplicar Sobel    
    sobel = AplicarSobel(img_bw, k = 5)
    sobel = aplicarThreshold(sobel, threshold_min = threshold_min, threshold_max = 255)
#    mostrarImagem(sobel)
    
    # Mascaras Combinadas
    mascara = cv2.bitwise_or(mascaraAmarelo, mascaraBranco)
    mascara = cv2.bitwise_or(mascara, sobel)

    return mascara


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
    # OBS
    # X = EIXO DA ALTURA DA IMAGEM (720)
    # Y = EIXO DA LARGURA DA IMAGEM (1280)
    # Achar picos do histograma da metade de baixo da imagem    
    minpix = 50
    histograma = np.sum(img[int(ALTURA/2):,:], axis=0)
    metade = int(LARGURA/2)
    picoEsquerdoBase = np.argmax(histograma[:metade])
    picoDireitoBase = np.argmax(histograma[metade:]) + metade    
    
    mascara = np.zeros_like(img)
    coordenadas = img.nonzero()
    coordenadasx = np.array(coordenadas[0])
    coordenadasy = np.array(coordenadas[1])
    
    faixaEsquerda = np.array([])
    faixaDireita = np.array([])
    
    # Para cada janela horizontal, gerar uma mascara
    ultimoPicoEsquerdo = picoEsquerdoBase
    ultimoPicoDireito = picoDireitoBase
    picoEsquerdo = picoEsquerdoBase
    picoDireito = picoDireitoBase
    alturaJanela = ceil(ALTURA/n_janelas)
#    cont=1    
    for inicioJanela_X in np.arange(0, ALTURA, alturaJanela):
        fimJanela_X = inicioJanela_X + alturaJanela 
#        print('Iteraçao %d' %cont)
#        print('Pico Esquerdo: %d' %picoEsquerdo)
#        print('Pico Direito: %d\n' %picoDireito)
#        cont += 1
        
        if (not checarPicoValido(picoEsquerdo, ultimoPicoEsquerdo, tolerancia)):
            picoEsquerdo = ultimoPicoEsquerdo
        
        inicioJanelaEsquerda_Y = picoEsquerdo-margem
        fimJanelaEsquerda_Y = picoEsquerdo+margem
#        mascara[inicioJanela_X:fimJanela_X, inicioJanelaEsquerda_Y:fimJanelaEsquerda_Y] = 1

        if (not checarPicoValido(picoDireito, ultimoPicoDireito, tolerancia)):
            picoDireito = ultimoPicoDireito
        inicioJanelaDireita_Y = picoDireito-margem
        fimJanelaDireita_Y = picoDireito+margem
#        mascara[inicioJanela_X:fimJanela_X, inicioJanelaDireita_Y:fimJanelaDireita_Y] = 1
        
        faixaEsquerdaIds = ((coordenadasy >= inicioJanelaEsquerda_Y) & (coordenadasy <= fimJanelaEsquerda_Y) &
                            (coordenadasx >= inicioJanela_X) & (coordenadasx >= fimJanela_X)).nonzero()[0]

        faixaDireitaIds = ((coordenadasy >= inicioJanelaDireita_Y) & (coordenadasy <= fimJanelaDireita_Y) &
                            (coordenadasx >= inicioJanela_X) & (coordenadasy >= fimJanela_X)).nonzero()[0]
        
        faixaDireita = np.append(faixaDireita, faixaDireitaIds, axis=0)
        faixaEsquerda = np.append(faixaEsquerda, faixaEsquerdaIds, axis=0)
        
        ultimoPicoEsquerdo = picoEsquerdo
        ultimoPicoDireito = picoDireito
        
        if len(faixaEsquerdaIds) >= minpix:
            picoEsquerdo = np.int(np.mean(coordenadasy[faixaEsquerdaIds]))
            
        if len(faixaDireitaIds) >= minpix:
            picoDireito = np.int(np.mean(coordenadasy[faixaDireitaIds]))
    
    faixaEsquerdax = coordenadasx[faixaEsquerda.astype(int)]
    faixaEsquerday = coordenadasy[faixaEsquerda.astype(int)] 
    faixaDireitax = coordenadasx[faixaDireita.astype(int)]
    faixaDireitay = coordenadasy[faixaDireita.astype(int)] 
    
    # Achar polinomio de segunda ordem (y = ax**2 + bx + c)
    faixaEsquerda_fit = np.polyfit(faixaEsquerdax, faixaEsquerday, 2)
    faixaDireita_fit = np.polyfit(faixaDireitax, faixaDireitay, 2)
    
    # Generate x and y values for plotting
#    plotx = np.linspace(0, img.shape[0]-1, img.shape[0] )
#    faixaEsquerda_fity = faixaEsquerda_fit[0]*plotx**2 + faixaEsquerda_fit[1]*plotx + faixaEsquerda_fit[2]
#    faixaDireita_fity = faixaDireita_fit[0]*plotx**2 + faixaDireita_fit[1]*plotx + faixaDireita_fit[2]

#    return mascara, faixaEsquerda_fity, faixaEsquerda_fit, faixaDireita_fity, faixaDireita_fit
    return faixaEsquerda_fit, faixaDireita_fit

def pintarPista(img, x, pontosRua, esq_fit, dir_fit, transparencia, cor):
    # Gerar Pontos    
    p_esq = np.poly1d(esq_fit)
    p_dir = np.poly1d(dir_fit)
    y_esq = p_esq(x)
    y_dir = p_dir(x)

    pts_esq = np.stack((y_esq,x),1)
    pts_dir = np.stack((y_dir,x),1)
    
    pts = np.append(pts_esq, np.flip(pts_dir,0),0).astype(np.int32)    
    
    # Gerar mascara da pista    
    origem = np.float32([[0,0], [LARGURA, 0], [LARGURA, ALTURA], [0, ALTURA]])
    matDist = cv2.getPerspectiveTransform(origem, pontosRua)
    
    mascaraPista = np.zeros([ALTURA, LARGURA, 3])
    mascaraPista = cv2.fillConvexPoly(mascaraPista, pts, np.multiply(cor, transparencia))
    mascaraPista = cv2.warpPerspective(mascaraPista, matDist, (LARGURA, ALTURA))
    
#    pistaColorida = mascaraPista * cor * transparencia
    
    # Pintar rua
    final = cv2.add(mascaraPista, img, dtype=0)        
    
    
    return final

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

# pontos para trcar a rua
n=5
intervalo = ceil(ALTURA / (n-1))
x = np.arange(0, ALTURA+1, intervalo)

# PIPELINE

start = time.time()
mascara = processarImagem(imagensTeste[0], pontosRua, 50)
t1=time.time()-start

start2 = time.time()
esq_fit, dir_fit = gerarCurvasDaFaixa(mascara, 5)
t2=time.time()-start2

start3 = time.time()
final = pintarPista(imagensTeste[0], x, pontosRua, esq_fit, dir_fit, 0.3, [255,255,0])
t3=time.time()-start3

end=time.time()
t=end-start
print('tempo total: %.4f s' %(t))
print('tempo f1: %.4f s' %(t1))
print('tempo f2: %.4f s' %(t2))
print('tempo f3: %.4f s' %(t3))
mostrarImagem(final)

mascara = processarImagem(imagensTeste[0], pontosRua, 50)
mostrarImagem(mascara)
cv2.imwrite('mascara.png', mascara)
start = time.time()
mascara2, out_img, esqy, esq_fit, diry, dir_fit = gerarCurvasDaFaixa(mascara, 7)
end=time.time()
t=end-start
print(t)
mostrarImagem(mascara2 * 255)
mostrarImagem(out_img)



