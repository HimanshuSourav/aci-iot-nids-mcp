# mcp_wrapper.py

import joblib

class MCPModel:
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_dict):
        # Convert input dict to DataFrame for preprocessing
        import pandas as pd
        X = pd.DataFrame([input_dict])
        X_proc = self.preprocessor.transform(X)
        return self.model.predict(X_proc)[0]

