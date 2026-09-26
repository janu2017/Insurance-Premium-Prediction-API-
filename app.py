#  Step 1 — activate environment
#PowerShell:
#  cd "C:\Users\JANHAVI\OneDrive\Desktop\Fast API Project"
# .\fast_env\Scripts\activate

#      Step i) — Start FastAPI
#        python -m uvicorn app:app --reload (Type this)
#        Keep this terminal running.
#        You should have:
#        http://127.0.0.1:8000


# Step 2 — Open another terminal
# Activate the same environment:
# cd "C:\Users\JANHAVI\OneDrive\Desktop\Fast API Project"
# .\fast_env\Scripts\activate

#     Step ii) — Start Streamlit
#           streamlit run frontend.py
#           Then open:
#           http://localhost:8501

# NOTE:there is one pdf in ML_model_frontain_...in same folder where you can find the detail of the project
#(In this project we are using only 3 files from this folder --app.py(Backend), frontain.py, model.pkl(which is generated after running fastapi_ml_model.py))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output,MODEL_VERSION,model





app = FastAPI()





#Human redable
@app.get('/')
def home():
    return{'message':'insurance_premium_predicton_API'}

#machine redable
@app.get('/health')
def health_check():
    return{
        'status': 'ok', 'model_version':MODEL_VERSION, 'model_loded': model is not None
    }

@app.post('/predict',response_model=PredictionResponse)#PredictionResponse is pydantic model's class imported from schema.prediction_response,here we validating the output data with pydantic class 

def predict_premium(data: UserInput):#UserInput is pydantic model's class imported from schema.user_inpute,here we converting the input data into pydantic class object

    user_input={
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    # Prediction
    try:
       prediction=predict_output(user_input)
       return JSONResponse(status_code=200,content={'Response': str(prediction)})

    except Exception as e:
       return JSONResponse(status_code=500,content=str(e))






