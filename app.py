from flask import Flask, request, url_for, render_template
import json
import glob
from PIL import Image


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html", value = "World")
    if request.method == 'POST':
        # print(request.get_json())
        stat = request.data
        # print(requet.POST.getlist())
        print(stat)
        # print(request)
        return "Hello"


if __name__ == "__main__":
    app.run(debug = True)
