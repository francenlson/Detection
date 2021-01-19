import cv2
#criando um classificador de imagens específico para frontal faces
# o arquivo .xml usado é que vai determinar o tipo de classificador = aqui um detector de faces frontal

classificador = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

#DEFININDO QUAL SERÁ A IMAGEM QUE SERÁ USADA PARA SER CLASSIFICADA
imagem = cv2.imread('pessoas\\faceolho.jpg')

#CONVERTENDO A IMAGEM EM TONS DE CINZA PARA MELHOR PROCESSAMENTO CONFORME DOC DO OPENCV
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#PROCESSAMENTO DA IMAGEM ** IMAGEM CINZA
facesDetectadas = classificador.detectMultiScale(imagemCinza)

#IMPRIMENDO QUANTAS FACES FORAM DETECTADAS
print(len(facesDetectadas))

#IMPRIMINDO A POSIÇÃO DE CADA FACE DETECTADA
print(facesDetectadas)

# **** CRIANDO O QUADRADINHO AO REDOR DE CADA FACE DETECTADA ****
for (x, y, l, a) in facesDetectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Faces Encontradas", imagem)
cv2.waitKey()




