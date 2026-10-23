# Bank Marketing — Term Deposit Prediction

Predicts which bank customers will subscribe to a term deposit, scored **before**
the call is made, so a campaign can be ordered by who is worth ringing first.

Live API: `hhttps://bank-marketing-api-3mst.onrender.com`
Interactive `docs: https://bank-marketing-api-3mst.onrender.com/docs`


## Result

| ROC-AUC | **0.798 ± 0.015** (5-fold CV) |
| Baseline (most frequent) | 0.500 |
| Accuracy | 0.902 — reported only as a footnote, see below |

At a 0.5 cutoff the model flags 371 customers and 247 of them subscribe:
a **67% hit rate against an 11% base rate**.



## Data

41,188 past marketing calls from a Portuguese bank, 20 features after cleaning.
Source: UCI Bank Marketing dataset.


## What had to be dealt with

**`duration` is leakage.** Call length is the strongest column in the dataset and
it is unusable — it only exists after the call has happened, and the prediction is
needed before. Including it lifts AUC to 0.955 and the model is worthless in
production. Dropped.

**Accuracy is the wrong metric here.** The target is 11.3% positive, so a model
that answers "no" to everyone scores 88.7%. ROC-AUC is reported instead.

**`pdays` uses 999 as a sentinel** for "never contacted before" — 96% of rows.
Left as-is, the model reads it as a measurement. Split into `contacted_before`
(a flag) and `pdays` (a real value, NaN where there isn't one).

**Missing data is stored as the string `"unknown"`**, so `isna()` reports zero
missing values across the whole table. 21% of `default` is affected.



## Model

`HistGradientBoostingClassifier` inside a `Pipeline` with one-hot encoding for the
10 categorical columns. Chosen over logistic regression because it handles NaN
natively, which `pdays` has by design after the fix above.

`handle_unknown="ignore"` on the encoder: a category arriving at prediction time
that wasn't in the training data leaves its columns at zero instead of raising.

Scored with 5-fold cross-validation rather than a single split — the single split
read 0.814, which is inside the CV spread but flattering.



## API

`POST /predict` — one customer in, subscription probability out.

```bash
curl -X POST https://YOUR-APP.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @customer.json
```

```json
{"will_subscribe": false, "probability": 0.0782}
```

The 0.5 cutoff is the default, not a tuned decision. Lowering it catches more
subscribers at the cost of more wasted calls; that trade-off belongs to whoever
is running the campaign.

Hosted on Render's free tier, which sleeps after 15 minutes idle — the first
request after that takes about a minute.

## Running it locally
