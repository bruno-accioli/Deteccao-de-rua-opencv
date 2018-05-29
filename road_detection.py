import numpy as np
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from math import floor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
    sobel = 255 * aplicarThreshold(sobel, threshold_min = threshold_min, threshold_max = 255)
#    mostrarImagem(sobel)
    
    # Mascaras Combinadas
    mascara = cv2.bitwise_or(mascaraAmarelo, mascaraBranco)
    mascara = cv2.bitwise_or(mascara, sobel)
#    mascara = cv2.bitwise_or(mascara, sobely)
#    mostrarImagem(mascara)
#    mostrarImagem(mascara)
    return mascara
#    res = cv2.bitwise_and(img_distorcida,img_distorcida, mask=mascara)
#    mostrarImagem(res)

#processarImagem(imagensTeste[0], pontosRua, 21)
#for i in range(len(imagensTeste)):
#    processarImagem(imagensTeste[i], pontosRua)
#    mostrarImagem(res)

#def AplicarMascaraDeCor(img, low, up)

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
#mostrarImagem(mascara)
#hist = np.histogram(mascara)
import time
km = KMeans(n_clusters = 2, max_iter=3)
start = time.time()
hist = mascara.sum(axis=0)
#plt.plot(hist)
teste = np.flatnonzero(mascara)%LARGURA
#plt.hist(a)

#km.fit(hist.reshape(-1,1))
km.fit(teste.reshape(-1,1))
end=time.time()
t=end-start
print(km.cluster_centers_)

start = time.time()
end=time.time()
t=end-start
print(t)
    


#BGR
#img = np.copy(imagensTeste[0])
#img = cv2.circle(img, tuple(topoEsq), 3, (255,0,0), thickness=1, lineType=8, shift=0) 
#img = cv2.circle(img, tuple(topoDir), 3, (0,255,0), thickness=1, lineType=8, shift=0)
#img = cv2.circle(img, tuple(baseDir), 3, (0,0,255), thickness=1, lineType=8, shift=0)
#img = cv2.circle(img, tuple(baseEsq), 3, (0,255,255), thickness=1, lineType=8, shift=0)
#
#mostrarImagem(img)



