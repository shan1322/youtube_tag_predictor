from flask import Flask, redirect, url_for
from markupsafe import escape
from flask import request
import json
import joblib
import os
from flask import render_template
from tag_recommender import TagRecommendation

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def get_form_data():
    if request.method == 'POST':
        title, similarity = list(request.form.values())
        if similarity == "":
            similarity = 0.7
        out = get_data(title, float(similarity))
        return render_template('home.html', check=1, text=out)
    else:
        return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_result():
    return render_template('home.html', check=1, text="bawasir12123123123")


def get_data(title, similarity=0.7):
    tag_rec = TagRecommendation(directory="data")
    tag_rec.load_data()
    # tag_rec.knn_pipe(mode="train", file="knn_tag.pkl")
    model = joblib.load(os.path.join("data", "knn_tag.pkl"))

    out = tag_rec.knn_pipe(mode="predict", file="knn_tag.pkl",
                           df_or_string=title, threshold=similarity, model=model)
    out = list(set(out))
    if len(out) == 0:
        return "no tag recommended change similarity and check"
    else:
        return ",".join(out)
if __name__ == "__main__":
    '''tag_rec = TagRecommendation(directory="data")
    tag_rec.load_data()
    tag_rec.knn_pipe(mode="train", file="knn_tag.pkl", model="", df_or_string="")
    '''
    app.run(debug=True)
