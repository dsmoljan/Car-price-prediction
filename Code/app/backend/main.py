import pickle

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware

from dto.car_information_request import CarInformationRequest
from car_dataset_validator import CarDatasetValidator
from model_wrapper import ModelWrapper
app = FastAPI()

app.add_middleware(CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],)


DATASET_INFO_PATH = "model_objects/categorical_features_dict.pkl"
MODEL_PATH = "model_objects/final_model_random_forest.pkl"
PREPROCESSING_DICT_PATH = "model_objects/preprocessing_dict.pkl"

@app.on_event("startup")
async def startup_event():
    """
    Initalize FastAPI server and the base model used for prediction.
    """
    model_wrapper = ModelWrapper(preprocessing_dict_path=PREPROCESSING_DICT_PATH, model_path=MODEL_PATH)

    # https://stackoverflow.com/questions/71298179/fastapi-how-to-get-app-instance-inside-a-router
    app.model = model_wrapper
    app.validator = CarDatasetValidator(DATASET_INFO_PATH)
    with open(DATASET_INFO_PATH, 'rb') as file:
        app.categorical_features_dict = pickle.load(file)

@app.get("/")
def read_root():
    return {"message": "Welcome to the used car prediction API. To get started, please read the API documentation on /docs."}

@app.get("/docs/allowed/carinformationrequest")
def get_allowed_values():
    return app.categorical_features_dict

@app.post("/api/predict")
async def do_predict(request: Request, car_information: CarInformationRequest):
    """
    Perform prediction on input data. The expected data format is defined by CarInformation request. It is necessary to
    provide values for all fields, otherwise the model throws an exception. To get a list of allowed values for each
    field of CarInformationRequest, please check /docs/allowed/carinformationrequest
    :param car_information: DTO object containing the information about the car
    :param request: HTTP request
    :return:
    """
    validation_errors = app.validator.validate(car_information)
    if len(validation_errors) != 0:
        raise HTTPException(status_code=400, detail=validation_errors)
    return {"car_price": app.model.predict(car_information)}


def validate():
    pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
