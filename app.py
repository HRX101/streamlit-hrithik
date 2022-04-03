from cv2 import CascadeClassifier
import streamlit as st
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
import numpy as np
import cv2
import av
from streamlit_webrtc import RTCConfiguration, webrtc_streamer,VideoTransformerBase
st.title("hi ")
frame_window=st.image([])
run=st.checkbox("run")
p12=""
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.load("model.tflearn")
cascade="haarcascade_frontalface_default.xml"


p9=0
face_detect=cv2.CascadeClassifier(cascade)
my_label=""
l=""
p=0
l=""
pas=False
p1=""
s12=1
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img= frame.to_ndarray(format="bgr24")
       
        img=cv2.flip(img,1)
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        face=face_detect.detectMultiScale(img_gray,1.4)
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.rectangle(frame, (200,40),(450,350),(255, 0, 255), 2)
            pr=img
            roi = img_gray[y:y+h,x:x+w]
        #roi = cv2.resize(roi, (200, 200))
  
           # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
            #detected_face= cv2.morphologyEx(detected_face, cv2.MORPH_CLOSE, kernel)
            detected_face = cv2.resize( roi,(50,50))
            detected_face = np.array(detected_face).reshape((-1,50,50,1))
        
       # label = np.argmax(model.predict(detected_face))
            conf=model.predict(detected_face)[0]
            idx=np.argmax(conf)
            
            confiedence="{:.2f}%".format(conf[idx]*100)
            model_out=model.predict(detected_face)
        #print(model_out)
            
            if np.argmax(model_out) == 0:
                my_label = 'happy'

            elif np.argmax(model_out)==1:
                my_label="sad"
                
            elif np.argmax(model_out)==2:
                my_label="angry"
            else:
                my_label = 'unkwnon'
            
            #print(my_label)
            cv2.putText(img,my_label,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            cv2.putText(img,str(p),(120,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
            cv2.putText(img,str(confiedence),(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
        #cv2.rectangle(frame, (200,40),(450,350),(255, 0, 255), 2)

        return img


webrtc_streamer(key="key", video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration(
                {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
            )
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
