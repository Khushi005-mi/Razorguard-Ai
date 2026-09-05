# RazorGuard AI

AI-powered merchant risk intelligence for detecting refund abuse before it becomes merchant loss.

RazorGuard analyzes transaction and refund behavior, generates an explainable risk score, and routes suspicious cases for appropriate action.

## How It Works
Transaction
    ↓
Behavioral Features
    ↓
ML Risk Model
    ↓
Risk Score
    ↓
Policy Engine
    ↓
ALLOW / MONITOR / FLAG FOR REVIEW
    ↓
Human Review + Audit Log
## Key Features
Behavioral refund-abuse detection
ML-based risk scoring
Explainable risk factors
Deterministic policy engine
Human-in-the-loop review
Transaction evaluation
Decision logging and auditability
Risk monitoring dashboard
Tech Stack

Frontend

Next.js
TypeScript
Tailwind CSS

Backend

Python
FastAPI
Scikit-learn

Data & ML

Transaction/refund behavioral features
Logistic Regression
Held-out model evaluation
Risk Levels
Risk	Action
LOW	ALLOW
MEDIUM	MONITOR
HIGH	FLAG FOR REVIEW

RazorGuard is a defensive risk-management prototype. High risk indicates that a transaction requires additional scrutiny; it does not by itself prove fraud or abuse.

Project Structure
razorguard-ai/
├── frontend/
├── src/
│   ├── api/
│   ├── risk_engine/
│   ├── features/
│   ├── modeling/
│   ├── ml/
│   ├── audit/
│   └── db/
├── data/
├── models/
├── tests/
├── notebooks/
└── requirements.txt
Run Locally
Backend
pip install -r requirements.txt

Start the FastAPI server using the project's configured entry point.

Frontend
cd frontend
npm install
npm run dev

## Open:

http://localhost:3000
Vision

RazorGuard starts with refund abuse detection and is designed to evolve into a broader AI risk intelligence layer for modern payments and autonomous financial systems.
