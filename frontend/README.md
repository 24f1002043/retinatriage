# RetinaTriage — Frontend

React (Vite) single-page UI for the RetinaTriage diabetic retinopathy screener.
It uploads a fundus image to the FastAPI backend and renders the predicted DR
stage, the full class-probability distribution, and the Grad-CAM overlay
side by side with the original image.

See the [project README](../README.md) for the full stack and setup.

## Run

```bash
npm install
npm run dev      # http://localhost:5173
```

The backend must be running (default `http://localhost:8000`) for analysis to
work. To point at a different backend, copy `.env.example` to `.env` and set
`VITE_API_BASE`.

## Scripts

| Command           | Description                          |
| ----------------- | ------------------------------------ |
| `npm run dev`     | Start the Vite dev server with HMR   |
| `npm run build`   | Production build into `dist/`        |
| `npm run preview` | Serve the production build locally   |
| `npm run lint`    | Lint with Oxlint                     |

## Structure

```
src/
├── App.jsx      # Upload, analysis, and results UI (single component tree)
├── App.css      # Component styles
├── index.css    # Global reset, fonts, dark theme base
└── main.jsx     # React entry point
```

## Backend contract

`POST {VITE_API_BASE}/predict` with `multipart/form-data` field `file`:

```json
{
  "stage": 2,
  "stage_label": "Moderate",
  "confidence": 0.87,
  "probabilities": { "No DR": 0.02, "...": 0.0 },
  "heatmap": "<base64 PNG>",
  "inference_ms": 143.2,
  "untrained_warning": false
}
```

`untrained_warning` is `true` when the backend has no trained weights loaded;
the UI surfaces a banner so demo output is never mistaken for a real result.
