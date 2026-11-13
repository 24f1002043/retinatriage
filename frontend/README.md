# RetinaTriage Frontend

React (Vite) UI for RetinaTriage. It uploads a fundus image to the FastAPI backend and shows the predicted DR stage, the class probabilities, and the Grad-CAM overlay next to the original image.

See the [main README](../README.md) for the full setup.

## Run

```bash
npm install
npm run dev      # http://localhost:5173
```

The backend needs to be running (default `http://localhost:8000`). To use a different backend, copy `.env.example` to `.env` and set `VITE_API_BASE`.

## Scripts

| Command           | Description                    |
| ----------------- | ------------------------------ |
| `npm run dev`     | dev server with HMR            |
| `npm run build`   | production build into `dist/`  |
| `npm run preview` | serve the production build     |
| `npm run lint`    | lint with Oxlint               |

## Structure

```
src/
├── App.jsx      upload, analysis and results UI
├── App.css      component styles
├── index.css    global styles and dark theme
└── main.jsx     entry point
```

## Backend response

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

`untrained_warning` is `true` when the backend has no trained weights. The UI then shows a banner.
