import flask
from flask import Flask, request, render_template, jsonify, request, abort
from fake_news_api import make_prediction

app = Flask(__name__)
# app = Flask('FakeNewsDetectorApp')

@app.route("/", methods=['POST','GET'])
def fake_or_real():

    return render_template('home.html')
    
@app.route("/process", methods=['POST','GET'])
def process_text():
    x_input = request.args.get('textstr')

    output = make_prediction(x_input)
    print(x_input,output)
    return render_template('layout.html', x_input=x_input,
                                 prediction=output)
    # print (x_input)
if __name__ == "__main__":

    app.run(debug=True)
# app.run(debug=True)