from flask import Flask, request, url_for, render_template, redirect
import numpy as np
import urllib
import torch
from PIL import Image
from torchvision import transforms
from model import get_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html", value="Output")
    if request.method == 'POST':
        print("Image recieved")
        data_url = request.values
        x = [i for i in data_url.items()]
        url = x[0][1]

        newimg = Image.open(urllib.request.urlopen(url))

        transform = transforms.Compose([transforms.Resize((32, 32)), 
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                       ])

        newimg = transform(newimg).unsqueeze(0)
        model = get_model()
        model.eval()
        with torch.no_grad():
            output = model(newimg)
        print(output)

        pred = output.argmax(dim=1).item()
        print("Predicted value : ", pred)

        return str(pred)

if __name__ == "__main__":
    app.run(debug = True)
