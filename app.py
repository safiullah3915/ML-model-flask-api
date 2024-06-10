from flask import Flask, abort, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

API_KEY = 'e77b532303397f75a312900293457ca9'

with open('BAI_score.pkl', 'rb') as file:
    anxiety_model = pickle.load(file)
with open('standard_score.pkl', 'rb') as file:
    anxiety_standard = pickle.load(file)
with open('ridge.pkl', 'rb') as file:
    depression_model = pickle.load(file)

@app.before_request
def before_request():
    key = request.headers.get('x-api-key')
    if not key or key != API_KEY:
        abort(401, description="Unauthorized: Incorrect or missing API key.")

def predict_anxiety(responses):
    responses = [responses.get(q, 0) for q in [
        "Numbness or tingling", "Feeling hot", "Wobbliness in legs", "Unable to relax", "Fear of the worst happening",
        "Dizzy or lightheaded", "Heart pounding/racing", "Unsteady", "Terrified or afraid", "Nervous", "Feeling of choking",
        "Hands trembling", "Shaky/unsteady", "Fear of losing control", "Difficulty in breathing", "Fear of dying",
        "Scared", "Indigestion", "Faint/lightheaded", "Face flushed", "Hot/cold sweats"
    ]]
    responses = np.array([responses])
    responses_scaled = anxiety_standard.transform(responses)
    predicted_score = anxiety_model.predict(responses_scaled)[0]
    level, resources = categorize_anxiety(predicted_score)
    return predicted_score, level, resources

def predict_depression(responses):
    response_values = [responses.get(q, 0) for q in responses.keys()]
    responses = np.array([response_values])
    predicted_score = depression_model.predict(responses)[0]
    level, resources = categorize_depression(predicted_score)
    return predicted_score, level, resources

def categorize_anxiety(score):
    if score <= 21:
        return 'Low Anxiety', [
            'https://www.youtube.com/watch?v=ZidGozDhOjg',
            'https://www.youtube.com/watch?v=VRxOmosteCc',
            'https://www.youtube.com/watch?v=8vfLmShk7MM'
        ]
    elif 22 <= score <= 35:
        return 'Moderate Anxiety', [
            'https://www.youtube.com/watch?v=_eWEGVE8f4w',
            'https://www.youtube.com/watch?v=HRkGYNZdlDw',
            'https://www.youtube.com/watch?v=JA86YOd4zx4'
        ]
    else:
        return 'Potentially Concerning Levels of Anxiety', [
            'https://www.youtube.com/watch?v=QLjPrNe63kk',
            'https://www.youtube.com/watch?v=MdHXlAgUe9Y'
        ]

def categorize_depression(score):
    if score <= 10:
        return 'Normal or no depression', [
            'https://www.youtube.com/watch?v=Bk0lzv8hEU8',
            'https://www.youtube.com/watch?v=TEwoWxLwCfA',
            'https://www.youtube.com/watch?v=OVJL850rAD8'
        ]
    elif 11 <= score <= 16:
        return 'Mild depression', [
            'https://www.youtube.com/watch?v=Y8qJ_0J2qKo',
            'https://www.youtube.com/watch?v=7sTWbgcuP2w',
            'https://www.youtube.com/watch?v=7DoQMnmo0v8'
        ]
    elif 17 <= score <= 20:
        return 'Borderline clinical depression', [
            'https://www.youtube.com/watch?v=qzTbEraKIOI',
            'https://www.youtube.com/watch?v=KSClXw4Wfxs'
        ]
    elif 21 <= score <= 30:
        return 'Moderate depression', [
            'https://www.youtube.com/watch?v=KSClXw4Wfxs',
            'https://www.youtube.com/watch?v=KSClXw4Wfxs'
        ]
    elif 31 <= score <= 40:
        return 'Severe depression', [
            'https://www.youtube.com/watch?v=KSClXw4Wfxs',
            'https://www.youtube.com/watch?v=KSClXw4Wfxs'
        ]
    else:
        return 'Extreme depression', [
            'https://www.youtube.com/watch?v=KSClXw4Wfxs',
            'https://www.youtube.com/watch?v=KSClXw4Wfxs'
        ]

        
@app.route('/', methods= ['POST','GET'])
@app.route('/predict', methods=['POST','GET'])
def mental_health():
    data = request.get_json()
    anxiety_score, anxiety_level, anxiety_resources = predict_anxiety(data['anxiety'])
    depression_score, depression_level, depression_resources = predict_depression(data['depression'])
    response = jsonify({
    'Anxiety': {
        'Score': anxiety_score,
        'Level': anxiety_level,
        'Resources': anxiety_resources
    },
    'Depression': {
        'Score': depression_score,
        'Level': depression_level,
        'Resources': depression_resources
    },
    'Physical_Exercise': {
        'Link': [
            "https://www.youtube.com/watch?v=fb3lDTS5IS4",
            "https://www.youtube.com/watch?v=6EysBiKaKmk",
            "https://www.youtube.com/watch?v=e9B3QWESkLI",
            "https://www.youtube.com/watch?v=LHLVgNBnFso",
            "https://www.youtube.com/watch?v=QevFo8wsXZ4"
        ]
    },
    'Reading_Articles': {
        'Link': [
            "https://communitymindset.org/",
            "https://peoplehouse.org/",
            "https://www.medicalnewstoday.com/articles/8933",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181767/",
            "https://www.healthline.com/health/can-you-cure-depression",
            "https://emedicine.medscape.com/article/286759-treatment?form=fpf",
            "https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2022.988648/full",
            "https://www.thelancet.com/journals/lanpsy/article/PIIS2215-0366(20)30036-5/fulltext"
        ]
    }
})
    return response

if __name__ == '__main__':
    app.run(debug=True)

