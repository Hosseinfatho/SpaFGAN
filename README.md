# Camera Position Visualization Project

This project displays OME-Zarr data using Vitessce, with a Flask backend and a React frontend.

## Project Structure

```
/
├── backend/           # Python Flask backend
│   ├── CameraPosition/  # Python Virtual Environment (created by user)
│   ├── static/        # Static files (if served by Flask - currently unused)
│   ├── templates/     # HTML templates (if served by Flask - currently unused)
│   ├── requirements.txt # Backend Python dependencies
│   └── server.py      # Flask application
├── frontend/          # React frontend
│   ├── node_modules/  # Node.js dependencies (managed by npm)
│   ├── public/        # Static assets for frontend
│   ├── src/           # React source code
│   ├── package.json   # Frontend dependencies and scripts
│   └── ...            # Other frontend config files (vite.config.js, etc.)
└── README.md        # This file
```

## Setup

1.  **Backend:**
    *   Navigate to the `backend` directory: `cd backend`
    *   Create/ensure the Python virtual environment exists (e.g., `python -m venv CameraPosition`)
    *   Activate the virtual environment: `.\CameraPosition\Scripts\activate` (Windows)
    *   Install dependencies: `pip install -r requirements.txt`

2.  **Frontend:**
    *   Navigate to the `frontend` directory: `cd frontend`
    *   Install dependencies: `npm install`

## Running the Application

1.  **Start Backend:**
    *   Open a terminal.
    *   Navigate to `backend/`.
    *   Activate the virtual environment: `.\CameraPosition\Scripts\activate`
    *   Run the server: `python server.py`
    *   The backend will run on `http://127.0.0.1:5000`.

2.  **Start Frontend:**
    *   Open a *second* terminal.
    *   Navigate to `frontend/`.
    *   Run the development server: `npm run dev`
    *   The frontend will likely run on `http://localhost:5173` (check terminal output).

3.  **View:**
    *   Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`). 