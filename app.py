from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from timeEstimator.Pipeline.prediction import Prediction


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def decode_prediction(hr, mi, se):
    hour, minute, second = hr, mi, se
    return f"{int(round(hour))}:{int(round(minute))}:{int(round(second))}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
           
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pred = Prediction()
            
            hr, mi, se = pred.start_prediction(file_path)
            updt_pred = decode_prediction(hr, mi, se)
            
            
            return render_template('result.html', 
                                   image_url=url_for('static', filename=f'uploads/{filename}'),
                                   predicted_time=updt_pred)
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
