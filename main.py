from flask import Flask,request,render_template
from predict import predict



app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files.get('image')
        if file:
            image_path=f"static/{file.filename}"
            file.save(image_path)
            prediction=predict(image_path)
            return render_template('index.html', prediction=prediction, image_path=image_path)
        else:
            return render_template('index.html', prediction=None, error="No file uploaded")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)        