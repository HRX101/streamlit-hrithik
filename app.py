from cv2 import CascadeClassifier
import streamlit as st
import cv2
import av
from streamlit_webrtc import RTCConfiguration, webrtc_streamer,VideoTransformerBase
st.title("hi ")
frame_window=st.image([])
run=st.checkbox("run")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img= frame.to_ndarray(format="bgr24")
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.flip(img,1)
        face=face_cascade.detectMultiScale(img_gray,1.4)
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        #cv2.rectangle(frame, (200,40),(450,350),(255, 0, 255), 2)

        return img


webrtc_streamer(key="key", video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration(
                {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
            )
)
