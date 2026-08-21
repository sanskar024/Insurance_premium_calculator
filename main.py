from fastapi import FastAPI
from fastapi.responses import JSONResponse
from user_input import UserInput
from typing import Literal, Annotated
import pickle
import pandas as pd

# import the ml model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
def root():
    return {"message": "Welcome to the Insurance Premium Category Predictor API!"}

@app.post('/predict')
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_probabilities = {cls: prob for cls, prob in zip(model.classes_, probabilities)}

        return JSONResponse(content={
            "response": {
                "predicted_category": prediction,
                "confidence": max(probabilities),
                "class_probabilities": class_probabilities
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




