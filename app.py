import os
import subprocess
from flask import Flask, request, redirect, render_template, jsonify, url_for

app = Flask(__name__)

@app.get('/')
def home():
	return render_template('index.html')

@app.post('/')
def get_file():
	file_names = os.listdir('./static')
	for f in file_names:
		os.remove(f'./static/{f}')

	if 'img' not in request.files:
		return redirect('/')
    
	file = request.files['img']

	if file and ('jpg' in file.filename):
		file.save(os.path.join('./static', file.filename))
		img_path = os.path.join('./static', file.filename)
	
	result = subprocess.run(
    [
        "venv\Scripts\python.exe",
        "./inference.py",
        "--ckpt_path",
        "saved_model\mobilenetv3_epoch=02_val_loss=0.27.ckpt",
        "--img_path",
        img_path,
    ],
    capture_output=True,
    text=True,
    )
	
	preds = result.stdout.replace("\n", "")
	
	return jsonify({
		'static' : url_for('static', filename=file.filename),
		'preds' : preds
	})

