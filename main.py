import cv2
import os
import numpy as np


#Recon é o reconhecimento facial padrão, ele que reconhece a imagem que vai ser exibida na camera.
def Recon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.1, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0:
        return None, None

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        graylist.append(gray[y:y + w, x:x + h])
        faceslist.append(faces[i])


    return graylist, faceslist

#Função predict serve para utilizar o face_recognizer, ele vai comparar a imagem do banco com a imagem atual
# na tela para verificar qual nome pode ser usado.

def predict(img):
    face, rect = Recon(img)

    if face is not None:
        for i in range(0, len(face)):
            label, confidence = face_recognizer.predict(face[i])
            print(confidence)
            if confidence < 100:
                label_text = banco[label]
                color = (0, 255, 0);
                (x, y, w, h) = rect[i]
                cv2.line(img, (x, y), (int(x + (w / 5)), y), color, 2)
                cv2.line(img, (int(x + ((w / 5) * 4)), y), (x + w, y), color, 2)
                cv2.line(img, (x, y), (x, int(y + (h / 5))), color, 2)
                cv2.line(img, (x + w, y), (x + w, int(y + (h / 5))), color, 2)
                cv2.line(img, (x, int(y + (h / 5 * 4))), (x, y + h), color, 2)
                cv2.line(img, (x, int(y + h)), (x + int(w / 5), y + h), color, 2)
                cv2.line(img, (x + int((w / 5) * 4), y + h), (x + w, y + h), color, 2)
                cv2.line(img, (x + w, int(y + (h / 5 * 4))), (x + w, y + h), color, 2)

                pt1 = (int(x + w / 2.0 - 150), int(y + h + 15))
                pt2 = (int((x + w / 2.0 + 50) + 90), int(y + h + 40))
                pt3 = (int(x + w / 2.0 - 120), int(y + h + (-int(y + h) + int(y + h + 20)) / 2 + 20))

                cv2.rectangle(img, pt1, pt2, (0, 0, 0), 1)
                cv2.putText(img, (label_text + "{:10.2f}".format((100 - confidence)) + "%"), pt3,
                            cv2.FONT_HERSHEY_PLAIN, 1.1, (150, 0, 0))

    return img

#Comparacao é a funcção que utilizei para reconhecer a imagem do banco e utilizar ela na Predict.

def Comparacao(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]

#Criação do nosso banco de imagens.
banco = [" ", "Guilherme", "Daniel", "Vinicius", "Lucas", "Gabriel", "Leo"]
# Foi deixado o primeiro slot em branco pois o como meu banco de fotos começa do 1 o slot "0" não poderia ter um nome.

#o Data() foi criado para ser a função que ira consultar as imagens que estão no banco, a serem utilizadas no futuro
#pela "predict".

def data():
    dirs = os.listdir("banco")

    faces = []
    labels = []

    for i in dirs:
        set = "banco/" + i

        label = int(i)

        for j in os.listdir(set):
            path = set + "/" + j
            img = cv2.imread(path)
            face, rect = Comparacao(img)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

#Criando tudo que é nescessário para o programa.
faces, labels = data()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
video_capture = cv2.VideoCapture(0)

#Linha para capturar a camera.

while True :
    ret, frame = video_capture.read()
    frame = predict(frame)
    cv2.imshow('Video', frame)
    cv2.waitKey(0)

'''
#Linha apara teste de imagem
img1 = cv2.imread('banco/2/1.jpg')
img = cv2.resize(img1, (600,600))
img = predict(img)
cv2.imshow('Image', img)
cv2.waitKey(0)
'''