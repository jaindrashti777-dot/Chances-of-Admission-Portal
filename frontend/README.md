# Admission Predictor Frontend

Next.js dashboard for the synthetic admission prediction demo. The UI collects a typed profile, calls the FastAPI backend, displays the prediction safely, and visualizes available feature contributions.

## Local Development

```bash
npm install
npm run dev
```

The app runs at `http://localhost:3000`. Start the backend at `http://localhost:8000` first, or set:

```env
NEXT_PUBLIC_API_URL=https://your-render-backend.onrender.com
```

## Scripts

```bash
npm run lint
npm run build
npm run start
```

## Deployment

Deploy this directory to Vercel with `frontend` as the project root. Set `NEXT_PUBLIC_API_URL` to the Render backend URL. The backend must allow the Vercel domain in `CORS_ORIGINS`.

## Scope

This frontend is a portfolio interface for a synthetic ML system. It should not be presented as real admissions advice or a production decision product.
