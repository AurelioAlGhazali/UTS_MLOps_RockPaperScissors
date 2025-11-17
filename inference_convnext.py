import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CKPT_PATH = Path("checkpoints_rps_real/convnext_tiny_rps_real_best.pth")


def load_model():
    """
    Loads ConvNeXt Tiny model + transforms using values saved in checkpoint.
    """
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH.resolve()}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    class_names = ckpt["class_names"]
    image_size  = ckpt["image_size"]
    mean        = ckpt["mean"]
    std         = ckpt["std"]
    num_classes = len(class_names)

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, tfm, class_names


@torch.no_grad()
def predict_image(img_path: str):
    """
    Predict Rock / Paper / Scissors from a single image path.
    """
    model, tfm, class_names = load_model()

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])

    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return top_label, top_conf, probs_dict

# Manual test
if __name__ == "__main__":
    test_img = r"C:\Users\ASUS\OneDrive\Documents\1. Aurelio\3. School\BINUS\1. Materi\3rd Semester\MLOps\UTS_MLOps\RockPaperScissor_Dataset\test\scissors\nasmi_214.png"
    label, conf, probs = predict_image(test_img)

    print(f"Prediction: {label} ({conf:.3f})")
    print("\nProbabilities:")
    for cls, p in probs.items():
        print(f"{cls:12s}: {p:.3f}")