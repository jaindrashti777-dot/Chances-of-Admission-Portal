# College Admission Predictor

An end-to-end machine learning portfolio project for a synthetic college admission prediction workflow. The repository includes offline data/model code, a FastAPI inference service, a Next.js dashboard, deployment configuration, and tests.

This is not an admissions decision tool. The dataset is synthetic and the target is generated from a formula, so model scores demonstrate engineering workflow rather than real-world predictive validity.

## Engineering Highlights

- FastAPI inference service with Pydantic request validation and structured error responses.
- Serialized scikit-learn pipeline loading for online prediction.
- Typed Next.js form that calls the backend through `NEXT_PUBLIC_API_URL`.
- SHAP-ready feature contribution output for explainability views.
- Split deployment model: Vercel frontend and Render backend.
- Lightweight Render API requirements separated from training/development dependencies.
- Pytest coverage for health, prediction, validation, metrics, and model availability.

## Architecture

```text
Synthetic data generator
        |
Training pipeline: preprocessing, feature engineering, model comparison
        |
models/pipeline.pkl
        |
FastAPI backend on Render: /health, /predict, /metrics, /api/chat
        |
Next.js frontend on Vercel: dashboard, methodology page, rule-based helper
```

Additional notes are in `docs/architecture.md`.

## Tech Stack

- Backend: Python 3.12, FastAPI, Pydantic, pandas, scikit-learn, SHAP
- ML: scikit-learn pipelines, XGBoost/LightGBM training options, joblib artifacts
- Frontend: Next.js App Router, TypeScript, CSS Modules, Recharts, lucide-react
- Testing: pytest, FastAPI TestClient, ESLint, Next.js production build
- Deployment: Docker, Render, Vercel

## Local Setup

```bash
git clone https://github.com/jaindrashti777-dot/Chances-of-Admission-Portal.git
cd Chances-of-Admission-Portal
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The repository includes `models/pipeline.pkl`. To regenerate synthetic data and retrain:

```bash
make generate
make train
```

Run the backend:

```bash
make api
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend defaults to `http://localhost:8000` when `NEXT_PUBLIC_API_URL` is unset.

## Environment Variables

Backend:

```env
PORT=8000
CORS_ORIGINS=http://localhost:3000,https://chances-of-admission-portal.vercel.app
```

Frontend:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, set `NEXT_PUBLIC_API_URL` in Vercel to the Render backend URL. Set `CORS_ORIGINS` in Render to include the Vercel production domain and localhost if needed for development.

## API Examples

Health:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Tenth_Percentage\":88.5,\"Twelfth_Percentage\":85,\"JEE_Percentile\":92.3,\"CUET_Score\":650,\"Category\":\"General\",\"State\":\"Maharashtra\",\"Family_Income\":12,\"Gender\":\"Male\",\"Gap_Year\":0,\"CGPA\":8.9,\"Backlogs\":0,\"Extracurricular\":1,\"Research_Paper\":1,\"Internship\":2,\"Desired_Branch\":\"CSE\",\"College_Tier\":\"Tier_1\"}"
```

Routes:

- `GET /` - API metadata
- `GET /health` - service health and model artifact availability
- `POST /predict` - admission score for one synthetic profile
- `GET /metrics` - local prediction log summary
- `POST /api/chat` - rule-based demo helper

## Deployment

Render backend:

1. Create a Render Web Service from this repository.
2. Use the included `render.yaml` and Dockerfile.
3. Health check path is `/health`.
4. Ensure `models/pipeline.pkl` is present in the repo or generated before deployment.
5. Set `CORS_ORIGINS` to the deployed Vercel URL.

Vercel frontend:

1. Import the repository into Vercel.
2. Set the root directory to `frontend`.
3. Set `NEXT_PUBLIC_API_URL` to the Render service URL.
4. Deploy.

## Testing

```bash
make test
cd frontend
npm run lint
npm run build
```

## Limitations / Model Card

- Dataset: synthetic, generated from configurable rules.
- Target: `Admission_Chance`, a generated value between 0 and 1.
- Intended use: demonstrate ML engineering, API integration, validation, deployment, and explainability patterns.
- Not intended for: real admissions decisions, counseling, ranking applicants, or policy analysis.
- Known limitation: high offline performance is expected because the model learns a synthetic generating process.
- Resume readiness score: transparent heuristic used for UI context, not a trained model output.

## License

MIT
