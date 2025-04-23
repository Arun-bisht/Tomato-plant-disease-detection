from flask import Flask, render_template,request,jsonify
#import model
#Load model here
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if request.method == "POST":
    # make prediction logic here and 
        return jsonify({"result": "success"})

app.run(debug=True, port=3000)
