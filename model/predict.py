import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f:#read binary
    model = pickle.load(f)

#MLflow
MODEL_VERSION='1.0.0'

class_labels=model.classes_.tolist() #['high','low','medium']

def predict_output(user_input:dict):

    df=pd.DataFrame([user_input])
    predicted_class=model.predict(df)[0]

    probabilities=model.predict_proba(df)[0]
    confidence=max(probabilities)

    class_probas=dict(zip(class_labels, map(lambda p:round(p,4), probabilities)))

    return{
        "predicted category":predicted_class,
        "confidence":round(confidence,4),
        "class_probabilities":class_probas
    }

    output=model.predict(input_df)[0]

    return output