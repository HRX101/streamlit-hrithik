import streamlit as st
import cv2
st.title("hi ")
frame_window=st.image([])
run=st.checkbox("run")
cap=cv2.VideoCapture(-1)

if cap.isOpened():
    while run:
        #st.write("hii")
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        frame_window.image(frame)

else:
    st.write("stopped")
