import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import streamlit as st


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CKPT_PATH = Path("checkpoints_rps_real/resnet18_rps_real_best.pth")


@st.cache_resource
def load_model_and_transform():

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH.resolve()}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    class_names = ckpt["class_names"]
    image_size  = ckpt["image_size"]
    mean        = ckpt.get("mean", [0.485, 0.456, 0.406])
    std         = ckpt.get("std",  [0.229, 0.224, 0.225])
    

    # ResNet18
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()


    # # ConvNext
    # mean        = ckpt["mean"]
    # std         = ckpt["std"]
    # num_classes = len(class_names)

    # try:
    #     weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    # except:
    #     from torchvision.models import ConvNeXt_Tiny_Weights
    #     weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

    # model = models.convnext_tiny(weights=weights)
    # in_features = model.classifier[2].in_features
    # model.classifier[2] = nn.Linear(in_features, num_classes)

    # model.load_state_dict(ckpt["model_state"])
    # model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, tfm, class_names


@torch.no_grad()
def predict_image_file(file, model, tfm, class_names):

    img = Image.open(file).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])

    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return img, top_label, top_conf, probs_dict


# UI
def main():

    st.set_page_config(page_title="RPS Classifier", page_icon="✋")
    st.title("Rock–Paper–Scissors Image Classifier (ResNet18)")
    st.write("Upload an image to classify it as **Rock**, **Paper**, or **Scissors**.")

    model, tfm, class_names = load_model_and_transform()

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:

        img, label, conf, probs = predict_image_file(uploaded, model, tfm, class_names)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence Score: `{conf:.4f}`")

        st.write("### Class Probabilities")
        st.bar_chart(probs)

        with st.expander("Raw Probabilities"):
            st.json(probs)




if __name__ == "__main__":
    main()
