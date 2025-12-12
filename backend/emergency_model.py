import torch
from torchvision import transforms, models
from PIL import Image

CKPT_PATH = "models/emergency_classifier_resnet18.pt"

_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_ckpt = torch.load(CKPT_PATH, map_location="cpu")
_class_names = _ckpt["class_names"]

_model = models.resnet18(weights=None)
num_feats = _model.fc.in_features
_model.fc = torch.nn.Linear(num_feats, len(_class_names))
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()


def classify_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    x = _tfms(img).unsqueeze(0)  # [1, C, H, W]
    with torch.no_grad():
        logits = _model(x)
        pred = logits.argmax(dim=1).item()
    return _class_names[pred]
