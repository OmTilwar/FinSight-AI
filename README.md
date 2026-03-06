# FinSight AI Assistant - Run Instructions

## Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **Model File**: Ensure `lendkraft_slm.gguf` is placed in `FinSight_Project/backend/models/`.

## Running the Backend (The Brain)
1. Navigate to the backend directory:
   ```bash
   cd FinSight_Project/backend
   ```
2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python main.py
   ```
   The backend will start at `http://localhost:8000`.

## Running the Frontend (The UI)
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd FinSight_Project/frontend
   ```
2. Install dependencies (if not already done):
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the link provided (usually `http://localhost:5173`) in your browser to chat with FinSight AI!
