from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from preprocessor import DataPreprocessor

# Load preprocessor and model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/random_forest.joblib")

# === Inference Input Schema ===
class InferenceInput(BaseModel):
    total_fwd_packets: int = Field(..., alias="Total Fwd Packets")
    total_backward_packets: int = Field(..., alias="Total Backward Packets")
    total_length_fwd_packets: float = Field(..., alias="Total Length of Fwd Packets")
    total_length_bwd_packets: float = Field(..., alias="Total Length of Bwd Packets")
    flow_duration: float = Field(..., alias="Flow Duration")
    
    # Optional categorical fields (you can expand this list as per actual dataset)
    protocol: Optional[str] = Field(default=None, alias="Protocol")
    service: Optional[str] = Field(default=None, alias="Service")

class InferenceOutput(BaseModel):
    prediction: float | list[float]

app = FastAPI(title="IoT Anomaly Detection Model", version="1.0.0")

@app.post("/predict", response_model=InferenceOutput)
def predict(input_data: InferenceInput):
    try:
        # Convert input to dataframe
        input_dict = input_data.dict(by_alias=True)
        df = pd.DataFrame([input_dict])

        # Let the preprocessor handle transformation
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
def get_metadata():
    return {
        "model_name": "IoT Anomaly Detector",
        "version": "1.0",
        "framework": "scikit-learn",
        "format": "joblib",
        "mcp_version": "1.0.0",
        "components": {
            "preprocessor": "StandardScaler with derived features",
            "model": "RandomForestClassifier"
        }
    }

@app.get("/schema")
def get_schema():
    return {
        "input_schema": {
            "Total Fwd Packets": "int",
            "Total Backward Packets": "int",
            "Total Length of Fwd Packets": "float",
            "Total Length of Bwd Packets": "float",
            "Flow Duration": "float",
            "Protocol": "optional str",
            "Service": "optional str"
        },
        "output_schema": {
            "prediction": "float or list of float"
        }
    }

