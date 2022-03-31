import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer,VideoTransformerBase
st.title("hi ")
frame_window=st.image([])
run=st.checkbox("run")
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame=cv2.Canny(img,100,200)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return av.VideoFrame.form_ndarray(img,format="bgr24")


webrtc_streamer(key="key", video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration(
                {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
            )
)
