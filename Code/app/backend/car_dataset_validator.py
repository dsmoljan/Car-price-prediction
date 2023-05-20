import pickle

from dto.car_information_request import CarInformationRequest


class CarDatasetValidator():
    """
    A validator for used car prices dataset.
    """

    def __init__(self, dataset_info_path: str):
        with open(dataset_info_path, 'rb') as file:
            self.categorical_values_dict = pickle.load(file)

    def validate(self, car_info_request: CarInformationRequest):
        """
        Checks if the passed car_info_request object's fields satisfy the constraints of the dataset, i.e if all
        the categorical fields have allowed values etc.
        :param car_info_request:
        :return: A list of validation errors
        """
        validation_errors = []
        for field_name, field_type in car_info_request.__annotations__.items():
            field_value = getattr(car_info_request, field_name)
            if field_type == str:
                allowed_values_list = self.categorical_values_dict.get(field_name, [])
                if field_value.lower() not in allowed_values_list and field_value.upper() not in allowed_values_list:
                    validation_errors.append(f"Invalid value for field '{field_name}': {field_value}")
            elif field_type == int:
                if field_value < 0:
                    validation_errors.append(f"Invalid value for field {field_name} - only positive integers allowed")

        return validation_errors
