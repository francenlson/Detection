import cv2


video = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# CAPTURA DE FOTOS PARA TREINAMENTO DE MÁQUINA
amostra = 1
numeroAmostra = 25
id = (input("DIGITE SEU IDENTIFICADOR : "))
largura, altura = 220, 220
print("Capturando as Faces ... ")



while True:
    conectado, imagem = video.read()
    #print(conectado)
    #print(frame)

    fotoCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(fotoCinza, scaleFactor=1.5, minSize=(100,100))
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                imagemFace = cv2.resize(fotoCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite("foto/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                print("[ Foto " + str(amostra) + " capturada com sucesso ... ]")
                amostra += 1

    cv2.imshow("Video", imagem)

    if (amostra >= numeroAmostra + 1):
        break

video.release()
cv2.destroyAllWindows()