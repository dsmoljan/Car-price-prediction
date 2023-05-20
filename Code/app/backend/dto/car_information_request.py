from datetime import datetime, date

from pydantic import BaseModel


class CarInformationRequest(BaseModel):
    year: int
    odometer: int
    # YYYY-MM-DD
    posting_date: date
    manufacturer: str
    condition: str
    cylinders: str
    fuel: str
    title_status: str
    transmission: str
    drive: str
    type: str
    paint_color: str







