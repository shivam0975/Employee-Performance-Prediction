import pickle
import numpy as np
from flask import Flask, request, render_template

model = pickle.load(open("best_model.pkl", "rb"))

app = Flask(__name__)

quarter_map = {"Quarter1": 0, "Quarter2": 1, "Quarter3": 2, "Quarter4": 3}
department_map = {"sweing": 0, "finishing": 1}
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/submit", methods=['POST'])
def submit():
    data = request.form

    quarter = int(data['quarter'])
    department = int(data['department'])
    day = int(data['day'])
    team = int(data['team'])
    targeted_productivity = float(data['targeted_productivity'])
    smv = float(data['smv'])
    over_time = int(data['over_time'])
    incentive = float(data['incentive'])
    idle_time = float(data['idle_time'])
    idle_men = int(data['idle_men'])
    no_of_style_change = int(data['no_of_style_change'])
    no_of_workers = float(data['no_of_workers'])
    month = int(data['month'])

    input_data = np.array([[quarter, department, day, team, targeted_productivity,
                            smv, over_time, incentive, idle_time, idle_men,
                            no_of_style_change, no_of_workers, month]])

    prediction = model.predict(input_data)[0]

    return render_template('submit.html', prediction=round(prediction, 3))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
