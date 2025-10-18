from streamlit_js_eval import streamlit_js_eval
import time
import os
from PIL import Image, ImageDraw
import streamlit as st
import sys
from torchvision.datasets import ImageFolder

st.write("Python running from:", sys.executable)
# Folders
st.set_page_config(
    page_title="Images classifcation Trainer", layout="centered")
st.title("Train me using any class you want to!")

base_image_folder = "dataset"
chosen_class = st.text_input("please enter type of class you want me to train",
                             placeholder="for example: person, girl, boy, sofa, dog, animal...")
container1, container2, container3 = st.columns([6, 5, 5])
with container1:
    with container1:

        uploaded_images = st.file_uploader("upload your photos using PNG. or JPG", type=[
            "png", "jpg", "jpeg"], accept_multiple_files=True, help="you can upload multiple images at once")


def message_before_saving():
    with st.spinner("Saving your images..."):
        time.sleep(1)


def no_null_field():
    if not chosen_class:
        # st.warning("Please enter the type!", icon="⚠️")
        streamlit_js_eval(
            js_expressions="alert('⚠️Please enter the type!')", key="popup-alert")
        return False
    elif not uploaded_images:
        st.warning("You have'nt upload any photo!", icon="⚠️")
        return False
    return True


exit_code = -1
with container2:
    if st.button("Train me!"):
        if no_null_field():
            with st.spinner("Saving images..."):
                train_path = os.path.join(base_image_folder, chosen_class)
                os.makedirs(train_path, exist_ok=True)

                for img_file in uploaded_images:
                    with open(os.path.join(train_path, img_file.name), "wb") as temp_file:
                        temp_file.write(img_file.getbuffer())

            st.toast(st.success(
                f"{len(uploaded_images)} images have been saved to {chosen_class} class"))
            with st.spinner("Training in progress..."):
                exit_code = os.system("python train.py")
        if exit_code == 0:
            st.success("✅ Training completed!")
        else:
            st.error("❌ Training failed. Check train.py.")

st.markdown("---")
st.header("Try it!")

test_image = st.file_uploader(
    "Upload an image to predict", type=["png", "jpg", "jpeg"])

if test_image:
    if st.button("Predict!"):
        if not test_image and os.path.exists("model.pth"):
            st.error("please upload an image to predict")
        else:
            try:
                from PIL import Image
                import torch
                from torchvision import transforms
                from model import SimpleCNN as base_cnn  # أو BaseCNN حسب اسم كلاك

    # تجهيز الصورة
                image = Image.open(test_image).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((1.0,), (1.5,))
                ])
                input_tensor = transform(image).unsqueeze(0)  # نضيف بعد batch
    # تحميل الأصناف
                if os.path.exists("dataset") and len(os.listdir("dataset")) > 0:
                    dataset = ImageFolder("dataset")
                    classes = dataset.classes

                else:
                    st.warning(
                        "⚠️ No dataset found or it's empty. Please train first.")
                    st.stop()

                num_classes = len(classes)

    # تحميل النموذج
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model = base_cnn(num_classes)
                model.load_state_dict(torch.load(
                    "model.pth", map_location=device))
                model.to(device)
                model.eval()

                with torch.no_grad():
                    iبnput_tensor = input_tensor.to(device)
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_class = classes[predicted.item()]

                st.image(image, caption="Uploaded Image",
                         use_container_width=True)
                st.success(f"✅ Prediction: `{predicted_class}`")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    # التنبؤ

#     message_before_saving()
#     if no_null_field():
#         train_path = os.path.join(base_image_folder, chosen_class)
#         os.makedirs(train_path, exist_ok=True)
#         for img_file in uploaded_images:
#             with open(os.path.join(train_path, img_file.name), "wb") as temp_file:
#                 temp_file.write(img_file.getbuffer())

#         st.toast(st.success(
#             f"{len(uploaded_images)} images have been saved to {chosen_class} class"))
