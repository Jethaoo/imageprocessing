import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
from flask import Flask, request, render_template, redirect, url_for
import os

# Initialize Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)  # background + apple
model.load_state_dict(torch.load("fasterrcnn_apples.pth", map_location="cpu"))
model.eval()

# Transform
transform = T.Compose([
    T.ToTensor()
])

def detect_apples(image, threshold=0.5):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)

    scores = outputs[0]["scores"]
    boxes = outputs[0]["boxes"]

    # Filter detections above threshold
    selected_boxes = boxes[scores > threshold].tolist()
    apple_count = len(selected_boxes)

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for box in selected_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return apple_count, image

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Detect apples
            image = Image.open(filepath).convert("RGB")
            count, result_image = detect_apples(image)

            # Save result image
            result_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
            result_image.save(result_path)

            return render_template("index.html", 
                                   count=count, 
                                   uploaded_image=url_for("static", filename="uploads/" + file.filename),
                                   result_image=url_for("static", filename="uploads/" + "result_" + file.filename))

    return render_template("index.html", count=None)

if __name__ == "__main__":
    app.run(debug=True)
