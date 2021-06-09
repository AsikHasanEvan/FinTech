from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from mobilenet_lib import detect_liveness_base64, detect_liveness

### init app ###
app = Flask(__name__, static_folder='static', static_url_path='/')

### enable cors for app ###
CORS(app)

### config ###
### app secret set ###
app.secret_key = "#livmob_5e"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MAX_CONTENT_PATH'] = 5242880 # 5mb in bytes
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

### declare routes ###
@app.route('/check-liveness', methods=['POST'])
def is_live():
    image_file = request.files['image']
    print(f"[INFO] Processing: {image_file.filename}")
    if image_file.filename != '':
        filename = str(uuid.uuid4()) + "_" + image_file.filename
        image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
        image_file.save(image_file_path)
        result = detect_liveness(image_file_path)
        os.remove(image_file_path)
        return jsonify({
            "filename": image_file.filename,
            "result": result
        })
    return jsonify({  
        "filename": None,
        "result": None
    })


@app.route('/check-liveness/base64', methods=['POST'])
def is_live_base64():
    try:
        image = request.get_json()["image"]

        result = detect_liveness_base64(image)
        
        response = {
            "statusCode": 200,
            "message": "Request Successful",
            "data": {
                "result": result
            }
        }
        return jsonify(response), 200

    except Exception as e:
        response = {
            "statusCode": 500,
            "message": str(e),
            "data": {
                "result": None
            }
        }
        return jsonify(response), 500
    


### starting app ###
if __name__ == '__main__':
    app.run("0.0.0.0", 3050, threaded=True, debug=True)

