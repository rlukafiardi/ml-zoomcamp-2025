from fastapi import FastAPI
import pickle
import uvicorn
from typing import Dict, Any

app = FastAPI(title='lead_scoring_prediction')

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post('/predict')
def predict(customer: Dict[str, Any]):
    result = pipeline.predict_proba(customer)[0, 1]
    return{
        "probability": float(result)
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9696)
