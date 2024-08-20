import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import sys
sys.path.append("/core/")
from core.pixel_wise_matching import pixel_wise_matching_l1
from core.pixel_wise_matching import pixel_wise_matching_l2
from core.window_base_matching import window_based_matching_l1
from core.window_base_matching import window_based_matching_l2
from core.window_base_matching import window_based_matching 

def compute_disparity_map(path_left_img, path_right_img, method):
    
    if method =="pixel-wise matching l1":
        disparity = pixel_wise_matching_l1(path_left_img,path_right_img, 16,save_result = False)
    
    elif method =="pixel-wise matching l2":
        disparity = pixel_wise_matching_l2(path_left_img,path_right_img, 16,save_result = False)

    elif method =="window-base matching l1":
        disparity_range = 64
        kernel_size = 3
        disparity = window_based_matching_l1(path_left_img,path_right_img, disparity_range, kernel_size,save_result = False)
    
    elif method =="window-base matching l2":
        disparity_range = 64
        kernel_size = 3
        disparity = window_based_matching_l2(path_left_img,path_right_img, disparity_range, kernel_size,save_result = False)

    elif method =="window-base matching cosine_similarity":
        disparity_range = 64
        kernel_size = 3
        disparity = window_based_matching(path_left_img,path_right_img, disparity_range, kernel_size,save_result = False)
    
    return disparity

st.title("Image Depth Estimation")

if 'disparity_map' not in st.session_state:
    st.session_state.disparity_map = None
col1, col2 = st.columns(2)

with col1:
    path_left_image = st.file_uploader("Upload Left Image", type=["jpg", "jpeg", "png"], key="left")
    if path_left_image:
        left_img = np.array(Image.open(path_left_image))
        st.image(left_img, caption="Left Image", use_column_width=True)

with col2:
    path_right_image = st.file_uploader("Upload Right Image", type=["jpg", "jpeg", "png"], key="right")
    if path_right_image:
        print(path_right_image.name)
        right_img = np.array(Image.open(path_right_image))
        st.image(right_img, caption="Right Image", use_column_width=True)

if path_left_image is not None and path_right_image is not None:
    col3, col4 = st.columns([1, 1])

    with col3:
        method = st.selectbox(
            "Select Disparity Computation Method",
            ["pixel-wise matching l1", "pixel-wise matching l2",
              "window-base matching l1", "window-base matching l2",
              "window-base matching cosine_similarity"]
        )
        if st.button("Click to Compute Dispatiry Map",
            ["Compute Disparity Map"]):
            st.session_state.disparity_map = compute_disparity_map(os.path.join("images",path_left_image.name), 
                                                    os.path.join("images",path_right_image.name), method)

   
       

if st.session_state.get("disparity_map") is not None:
    col3, col4 = st.columns([1, 1])
    with col3:
        st.image(st.session_state.disparity_map, caption="Disparity Map", use_column_width=True)

    with col4:
        color = cv2.cvtColor(cv2.applyColorMap(st.session_state.disparity_map,cv2 .COLORMAP_JET ), cv2.COLOR_BGR2RGB)
        st.image(color, caption="Disparity Map Color", use_column_width=True)