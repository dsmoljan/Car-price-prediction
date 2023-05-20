import pickle
import pandas as pd
from dto.car_information_request import CarInformationRequest

categorical = ['manufacturer', 'condition', 'cylinders',
               'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']

class ModelWrapper():
    def __init__(self, preprocessing_dict_path, model_path):
        with open(preprocessing_dict_path, 'rb') as file:
            self.preprocessing_dict = pickle.load(file)
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        assert self.model is not None
        assert self.preprocessing_dict is not None

    def predict(self, car_info_request: CarInformationRequest):
        age = car_info_request.posting_date.year - car_info_request.year
        data_dict = car_info_request.dict(exclude={"year", "posting_date"})
        data_dict["age"] = age
        df = pd.DataFrame(data_dict, index=[0])
        df[['age']] = self.preprocessing_dict["age_scaler"].transform(df[['age']])
        df[['odometer']] = self.preprocessing_dict["odometer_scaler"].transform(df[['odometer']])
        oh_enc = self.preprocessing_dict["one_hot_encoder"]
        oh_cols_train = pd.DataFrame(oh_enc.transform(df[categorical]))
        oh_cols_train.index = df.index
        df = df.drop(categorical, axis=1)
        df = pd.concat([df, oh_cols_train], axis=1)
        price = self.preprocessing_dict["price_scaler"].inverse_transform(self.model.predict(df.values).reshape(-1,1))[0][0]
        return price


