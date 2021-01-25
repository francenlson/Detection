import cv2
#criando um classificador de imagens específico para frontal faces
classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

#criando um classificador de imagens específico para OS OLHOS
classificadorOlhos = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')


imagem = cv2.imread('pessoas\\beatles.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificadorFace.detectMultiScale(imagemCinza)

for (x, y, l, a) in facesDetectadas:
    # RETANGULO NA FACE
    imagem - cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 0), 2)

    #usando a imagem da face para detectar os olhos
    regiao = imagem[y:y + a, x:x + l]

    # CONVERTENDO A IMAGEM "REGIÃO" PARA CINZA
    regiaoCinzaOlhos = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

    # PROCESSANDO A DETECÇÃO DOS OLHOS
    olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlhos, scaleFactor=1.08, minNeighbors=3)

    print(olhosDetectados)

    # RETANGULO NOS OLHOS
    for (ox, oy, ol, oa) in olhosDetectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
        

cv2.imshow("Faces e Olhos Detectados", imagem)
cv2.waitKey()


