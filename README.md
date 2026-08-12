# Modelwise

**Live site:** [Open Modelwise](https://modelwise-advisor.neglected-turtle.chatgpt.site)

Modelwise is an uncertainty-aware model recommender for tabular classification datasets.

Upload a CSV, choose the target column, and receive a calibrated shortlist from:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Histogram Gradient Boosting

## Performance

The frozen selector was evaluated on 32 newly downloaded datasets that were not used for training or tuning.

| Metric | Result |
|---|---:|
| Mean score regret | 0.90 percentage points |
| Median score regret | 0.00 percentage points |
| Within one point of best | 81.25% |
| Top-three coverage | 84.38% |
| Risk-model AUC | 0.731 |

## Run locally

Install the Python service:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uvicorn server:app --port 8000
```

In a second terminal, start the interface:

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

On macOS or Linux, replace `.venv\Scripts\` with `.venv/bin/`.
## Deploy

Deploy the API from the repository root with Render using `render.yaml`. The interface is published with OpenAI Sites; set `NEXT_PUBLIC_API_URL` to the Render service URL in the hosted environment.


## Python usage

```python
import pandas as pd

from model_selector import ModelSelector

frame = pd.read_csv("dataset.csv")
selector = ModelSelector()
result = selector.recommend(frame, target_column="target")

print(result)
```

## Decision policy

| Calibrated risk | Output |
|---|---|
| Up to 0.10 | Top model |
| 0.10 to 0.25 | Top two models |
| Above 0.25 | Top three models |

Recommendations should be confirmed on a project-specific validation set.

## Structure

```text
artifacts/         trained selector, calibrator, metrics
web/               responsive upload interface
model_selector.py  inference and probing
server.py          upload API
requirements.txt   Python dependencies
```
