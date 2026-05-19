# Leaf Disease Area Detector

## Deployment on Render

1. Upload all project files to GitHub
2. Add your trained model:
   leaf_disease_unet_resnet34.pth
3. Open Render
4. Create New Web Service
5. Connect GitHub repository
6. Deploy

## Required Files

- app.py
- requirements.txt
- runtime.txt
- render.yaml
- templates/index.html
- leaf_disease_unet_resnet34.pth

## Run Locally

pip install -r requirements.txt
python app.py
