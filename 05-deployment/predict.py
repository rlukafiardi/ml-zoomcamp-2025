import pickle

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

data = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

result = pipeline.predict_proba(data)[0, 1]
print(f'The probability of given lead will convert is {result:.3f}')