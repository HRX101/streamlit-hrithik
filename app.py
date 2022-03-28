import streamlit as st
import cv2
st.title("hi ")
frame_window=st.image([])
run=st.checkbox("run")
cap=cv2.VideoCapture(cv2.CAP_V4L2)
camera_backends = cv2.videoio_registry.getCameraBackends()
print(camera_backends)
print([
    cv2.videoio_registry.getBackendName(apipref)
    for apipref in camera_backends
])
if cap.isOpened():
    while run:
        #st.write("hii")
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        frame_window.image(frame)

else:
    st.write("stopped")
