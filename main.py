from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("bank_model.pkl")
    yield


app = FastAPI(lifespan=lifespan, title="Bank marketing API")


class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(c: Customer):
    row = {
        "age": c.age,
        "job": c.job,
        "marital": c.marital,
        "education": c.education,
        "default": c.default,
        "housing": c.housing,
        "loan": c.loan,
        "contact": c.contact,
        "month": c.month,
        "day_of_week": c.day_of_week,
        "campaign": c.campaign,
        "pdays": None if c.pdays == 999 else c.pdays,
        "previous": c.previous,
        "poutcome": c.poutcome,
        "emp.var.rate": c.emp_var_rate,
        "cons.price.idx": c.cons_price_idx,
        "cons.conf.idx": c.cons_conf_idx,
        "euribor3m": c.euribor3m,
        "nr.employed": c.nr_employed,
        "contacted_before": 0 if c.pdays == 999 else 1,
    }

    prob = float(app.state.model.predict_proba(pd.DataFrame([row]))[0, 1])

    return {"will_subscribe": prob >= 0.5, "probability": round(prob, 4)}
