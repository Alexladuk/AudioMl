from flask import Flask, render_template, request
import pickle 

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        email_text = request.form.get('email-content')
    tokenized_text = tokenizer.transform([email_text])
    prediction = model.predict(tokenized_text)
    predictions = 1 if prediction==1 else -1
    return render_template('index.html', predictions = predictions, email = email_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

    