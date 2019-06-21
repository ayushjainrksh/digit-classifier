from flask import Flask, request, url_for, render_template
import numpy as np
import urllib
import torch
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html", value = "World")
    if request.method == 'POST':
        print("Image recieved")
        data_url = request.values
        x = [i for i in data_url.items()]
        url = x[0][1]

        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        print(image)
        img = torch.from_numpy(image).float()

        print(img.shape)

        return "Hello"


if __name__ == "__main__":
    app.run(debug = True)
