import gradio as gr
import timm, torch, numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
names = {
    "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis", "df": "Dermatofibroma",
    "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion"
}

m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7).to(device)
m.load_state_dict(torch.load("E2.pth", map_location=device))
m.eval()

cam = GradCAM(model=m, target_layers=[m.conv_head])
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def predict(img):
    if img is None:
        return None, None
    img = Image.fromarray(img).convert("RGB").resize((224, 224))
    rgb = np.array(img) / 255.0
    x = torch.tensor(((rgb - mean) / std).transpose(2, 0, 1)[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0].cpu().numpy()
    top = int(p.argmax())
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    overlay = show_cam_on_image(rgb.astype(np.float32), heat, use_rgb=True)
    return {names[c]: float(p[i]) for i, c in enumerate(classes)}, overlay

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload dermoscopy image"),
    outputs=[gr.Label(num_top_classes=7, label="Prediction"), gr.Image(label="Grad-CAM heatmap")],
    title="Skin Lesion Analysis (Research Prototype)",
    description="Not a medical device. For research and education only. Consult a dermatologist for any skin concern.",
)

import os
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
