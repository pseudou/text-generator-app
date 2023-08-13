from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from Settings import settings as default
from Settings import dataset
import os
from Model import model as net

# enabling CORS
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# load the pretrained model
model = net.load_model(path=os.path.join(default.WEIGHTS_PATH,dataset.WEIGHTS))
model.to(default.DEVICE)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/generate/', methods=['POST'])
@cross_origin()
def generate_bootstrap():
    if request.method == 'POST':
        # gather the required data 
        initial_text = request.args.get('initial_text')
        length = int(request.args.get('length'))

        # generate text
        generated = net.generate_text(model,initial_text,length)

    return jsonify({
        'input':initial_text,
        'lenght':length,
        'output':generated,
    })

if __name__=="__main__":
    app.run(debug=False, host= '0.0.0.0')