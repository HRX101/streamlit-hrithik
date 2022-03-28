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

        return img


webrtc_streamer(key="sample", video_transformer_factory=VideoTransformer)
