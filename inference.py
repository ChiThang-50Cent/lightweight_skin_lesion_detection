import argparse
import torch

from PIL import Image
from models import Models
from transforms import get_val_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classes", "-n", type=int, help="Number of classes", default=7
    )
    parser.add_argument(
        "--model_name",
        help="Choose model efficientnet, mobilenetv3, shufflenetv2",
        default="mobilenetv3",
    )
    parser.add_argument("--img_path", help="Path to image")
    parser.add_argument("--ckpt_path", help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Models(
        model_name=args.model_name,
        num_classes=args.num_classes,
        feature_extract=False,
        pre_trained=False,
    ).get_model()

    transform = get_val_transform(args.img_size)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model_weights = checkpoint["state_dict"]

    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)

    model.to(device)
    model.eval()

    images = Image.open(args.img_path).convert('RGB')
    # import numpy as np
    # print(np.array(images).shape)
    images = transform(images)

    with torch.no_grad():
        images = images.to(device)
        images = images.unsqueeze(0)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]

    label_map = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    detail_map = {
        "nv": "Melanocytic nevi",
        "mel": "dermatofibroma",
        "bkl": "Benign keratosis-like lesions ",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }
    print(detail_map[label_map[prediction.item()]])
