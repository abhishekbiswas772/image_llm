import streamlit as st
from PIL import Image
from main import main2, main1, main3, text_to_image_pipeline
import tempfile
from streamlit_option_menu import option_menu

selected = option_menu("Menu", ["Image to Image Search", 'Text to Image Search'], default_index=0)
if selected == "Image to Image Search":
    st.title("Image Upload and Display with Save Path")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.header("User Query Image")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file.name)
            saved_image_path = temp_file.name
            result = main2(file_path=saved_image_path)
            probable_part_number = result[-1]
            probable_image = result[0]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', width=300)
        with col2:
            st.header("Predicted Image From Milvus")
            image_predicted = Image.open("./base_images/" + probable_image)
            st.image(image_predicted, caption="Predicted Image From Milvus", width=300)
# elif selected == "Text to Image Search (summary based)":
#     st.title("Text to Image Search using summary")
#     user_input = st.text_input("Enter the text:")
#     if st.button('Submit'):
#         if user_input != "" or user_input is not None:
#             result_from_text_input = main3(user_input)
#             image_predicted = Image.open("./base_images/" + result_from_text_input)
#             st.image(image_predicted, caption="Predicted Image From Milvus", width=300)
elif selected == "Text to Image Search":
    st.title("Text to Image Search")
    user_input = st.text_input("Enter the text:")
    if st.button('Submit'):
        if user_input != "" or user_input is not None:
            result = text_to_image_pipeline(
                user_input
            )
            print(result)
            image_predicted = Image.open(result)
            st.image(image_predicted, caption="Predicted Image", width=300)




st.sidebar.header("Insert Images to Milvus")
if st.sidebar.button("Start Insertion Data"):
    main1()
