from flask import Flask, render_template, request
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

app = Flask(__name__)

# =====================================
# CONFIG
# =====================================

IMG_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# CREATE MODEL
# =====================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

# =====================================
# REBUILD MODEL FROM PART FILES
# =====================================

MODEL_FILE = "leaf_disease_unet_resnet34.pth"

part_files = sorted([
    f for f in os.listdir()
    if ".part" in f
])

print("Found model parts:")
print(part_files)

# merge all parts
with open(MODEL_FILE, "wb") as outfile:

    for part in part_files:

        print("Merging:", part)

        with open(part, "rb") as infile:
            outfile.write(infile.read())

print("Model reconstructed successfully!")

# =====================================
# LOAD MODEL WEIGHTS
# =====================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully!")

# =====================================
# IMAGE TRANSFORM
# =====================================

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# =====================================
# HOME ROUTE
# =====================================

@app.route("/", methods=["GET", "POST"])
def home():

    severity = None
    image_path = None
    mask_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            # create static folder
            os.makedirs("static", exist_ok=True)

            # save uploaded image
            image_path = os.path.join(
                "static",
                file.filename
            )

            file.save(image_path)

            # =====================================
            # READ IMAGE
            # =====================================

            img = Image.open(image_path).convert("RGB")

            img_np = np.array(img)

            # =====================================
            # PREPROCESS
            # =====================================

            aug = transform(image=img_np)

            img_tensor = aug["image"].unsqueeze(0).to(device)

            # =====================================
            # PREDICT
            # =====================================

            with torch.no_grad():

                pred = model(img_tensor)

                pred_mask = (
                    torch.sigmoid(pred) > 0.5
                ).float().cpu().squeeze().numpy()

            # =====================================
            # DISEASE %
            # =====================================

            severity = round(
                pred_mask.mean() * 100,
                2
            )

            # =====================================
            # SAVE MASK IMAGE
            # =====================================

            mask_img = (
                pred_mask * 255
            ).astype(np.uint8)

            mask_path = os.path.join(
                "static",
                "mask.png"
            )

            Image.fromarray(mask_img).save(mask_path)

    return render_template(
        "index.html",
        severity=severity,
        image_path=image_path,
        mask_path=mask_path
    )

# =====================================
# START APP
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )