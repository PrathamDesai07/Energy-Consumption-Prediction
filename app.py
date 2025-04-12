from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

import pandas as pd
from typing import Optional
from datetime import datetime
import joblib
import numpy as np

# Importing constants and pipeline modules from the project
from src.utils.main_utils import load_object
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import EnergyData, EnergyDataClassifier
from src.pipline.training_pipeline import TrainPipeline
from src.entity.artifact_entity import DataTransformationArtifact

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = None
class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    Datetime,Temperature,Humidity,WindSpeed,GeneralDiffuseFlows,DiffuseFlows
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Datetime: Optional[datetime] = None
        self.Temperature: Optional[float] = None
        self.Humidity: Optional[float] = None
        self.WindSpeed: Optional[float] = None
        self.GeneralDiffuseFlows: Optional[float] = None
        self.DiffuseFlows: Optional[float] = None
        
        
                
    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Datetime = form.get("Datetime")
        self.Temperature = form.get("Temperature")
        self.Humidity = form.get("Humidity")
        self.WindSpeed = form.get("WindSpeed")
        self.GeneralDiffuseFlows = form.get("GeneralDiffuseFlows")
        self.DiffuseFlows = form.get("DiffuseFlows")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "energydata.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()    
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    Datetime,Temperature,Humidity,WindSpeed,GeneralDiffuseFlows,DiffuseFlows
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()
        parsed_datetime = datetime.strptime(form.Datetime, "%m/%d/%Y %H:%M")
        formatted_datetime = int(parsed_datetime.strftime("%Y%m%d%H"))
        Energy_data = EnergyData(
                                Datetime= formatted_datetime,
                                Temperature = form.Temperature,
                                Humidity = form.Humidity,
                                WindSpeed = form.WindSpeed,
                                GeneralDiffuseFlows = form.GeneralDiffuseFlows,
                                DiffuseFlows = form.DiffuseFlows,
                                )

        # Convert form data into a DataFrame for the model
        energy_df = Energy_data.get_energy_input_data_frame()
        energy_df = energy_df.set_index('Datetime')
        energy_df.index = pd.to_datetime(energy_df.index)
        energy_df = Energy_data._create_features(df = energy_df)
        # Initialize the prediction pipeline
        model_predictor = EnergyDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=energy_df)[0]
        
        pipeline = joblib.load("./preprocessing_target.pkl")

        # Access the ColumnTransformer
        column_transformer = pipeline.named_steps['Preprocessor']

        # Access the MinMaxScaler inside it
        scaler = column_transformer.named_transformers_['MinMaxScaler']

        normalized_prediction = np.array([value])

        pred_values = scaler.inverse_transform(normalized_prediction)
        zone1, zone2, zone3 = pred_values[0]
        print(zone1, zone2, zone3)
        formatted_html = f"""
        Zone 1: {zone1:,.2f} kWh
        Zone 2: {zone2:,.2f} kWh
        Zone 3: {zone3:,.2f} kWh
        """
        print(formatted_html)
        status = f'Predicted consumption {type(pred_values)}'

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "energydata.html",
            {"request": request, "context": formatted_html},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)