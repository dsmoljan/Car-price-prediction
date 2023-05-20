import pickle

import requests
import streamlit as st
from pydantic import BaseModel

DATASET_INFO_PATH = "categorical_features_dict.pkl"

st.title("Car prediction app")

# when running from outside docker, replace api:8080 with localhost:8080
ENDPOINT_URL= f'http://api:8080/api/predict'


class CarInformationRequest(BaseModel):
    year: int
    odometer: int
    posting_date: str
    manufacturer: str
    condition: str
    cylinders: str
    fuel: str
    title_status: str
    transmission: str
    drive: str
    type: str
    paint_color: str


def send_to_backend(car_info):
    car_info_json = car_info.json()
    print(car_info_json)
    response = requests.post(ENDPOINT_URL, data=car_info_json)

    if response.status_code == 200:
        prediction = response.json()
        pred_price = prediction["car_price"]
        return pred_price
    else:
        st.error("Error occurred during prediction.")


def main():
    categorical_features_dict = None

    with open(DATASET_INFO_PATH, 'rb') as file:
        categorical_features_dict = pickle.load(file)
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=1930, max_value=2023)
        odometer = st.number_input("Odometer")
        posting_date = st.date_input("Posting Date (YYYY-MM-DD)")
        manufacturer = st.selectbox("Manufacturer", categorical_features_dict.get("manufacturer"))
        condition = st.selectbox("Condition", categorical_features_dict.get("condition"))
        cylinders = st.selectbox("Cylinders", categorical_features_dict.get("cylinders"))
    with col2:
        fuel = st.selectbox("Fuel", categorical_features_dict.get("fuel"))
        title_status = st.selectbox("Title Status", categorical_features_dict.get("title_status"))
        transmission = st.selectbox("Transmission", categorical_features_dict.get("transmission"))
        drive = st.selectbox("Drive", categorical_features_dict.get("drive"))
        car_type = st.selectbox("Car Type", categorical_features_dict.get("type"))
        paint_color = st.selectbox("Paint Color", categorical_features_dict.get("paint_color"))

    # Submit button
    if st.button("Submit"):
        car_info = CarInformationRequest(
            year=year,
            odometer=odometer,
            posting_date=str(posting_date),
            manufacturer=manufacturer,
            condition=condition,
            cylinders=cylinders,
            fuel=fuel,
            title_status=title_status,
            transmission=transmission,
            drive=drive,
            type=car_type,
            paint_color=paint_color
        )
        pred_price = send_to_backend(car_info)
        st.info(f"Predicted car price is {float(pred_price):.2f}$")


if __name__ == "__main__":
    main()