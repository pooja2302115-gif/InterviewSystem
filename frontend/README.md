# Interview Lab Frontend

Phase 17 provides the React chatbot client for the FastAPI backend.

## Run

Install dependencies and start Vite:

```bash
npm install
npm run dev
```

The development server proxies `/api` to `http://localhost:8000` by default. This avoids browser `localhost` and CORS problems in forwarded/container environments. Set `VITE_API_URL` only when the frontend must call another backend directly:

```bash
VITE_API_URL=https://your-api.example.com npm run dev
```

The backend must be running separately with `uvicorn backend.app:app --reload --port 8000`.

## Current surface

- Technical interview, DSA practice, and HR interview modes.
- Session-aware chat through `POST /chat`.
- New-session reset, starter prompts, loading state, errors, and copy response action.
- Responsive desktop sidebar and mobile navigation drawer.

The model remains a small educational checkpoint-backed system. The React app does not claim production-grade response quality.
