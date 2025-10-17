from pydantic import BaseModel, confloat


class Metrics(BaseModel):
    cci_mean: confloat(ge=0.0, le=1.0)
    hazard_mean: confloat(ge=0.0, le=1.0)
    survival_mean: confloat(ge=0.0, le=1.0)
