# 📈 IPO Success Predictor

> A full-stack AI-powered web application that predicts whether an IPO will list at a **gain or a loss** on the stock exchange — built for retail investors who want data-driven decisions.

---

## 📌 Project Overview

**IPO Success Predictor** is a B.Tech Data Science Smart Group Project (SGP) that combines machine learning with a modern web interface to help investors make smarter IPO application decisions. The system analyzes historical IPO subscription data and uses a trained Logistic Regression model to predict listing success with high accuracy.

The platform supports user accounts, prediction history tracking, a 12-chart analytics dashboard, a searchable IPO listings database, and smart insights — all wrapped in a dark fintech-themed UI.

---

## ❓ Problem Statement

Every year, thousands of IPOs are listed on Indian stock exchanges (NSE/BSE). While some deliver exceptional listing gains, many list below the issue price — causing retail investors to lose money. Most investors base their decisions on:

- Social media hype
- Broker recommendations
- News articles with no quantitative backing

There is no free, simple, data-driven tool for retail investors to predict IPO performance before applying.

---

## 🎯 Objectives

1. Build a machine learning model that predicts IPO listing outcome (Success / Fail)
2. Provide an easy-to-use web interface with IPO name auto-fill and auto-prediction
3. Show prediction probability, risk level, confidence level, and smart insights
4. Enable users to track their prediction history
5. Provide an analytics dashboard to explore IPO trends across sectors
6. Build a searchable, filterable IPO listings database with real date-based status tracking

---

## ✨ Features

### 🔍 Predictor
- Search IPO by name with live auto-suggestions
- Auto-fill all subscription fields from database
- Predict IPO success using a trained ML model
- Display Success / Failure probability bars
- Show IPO Score (0–100), Risk Level, Confidence Level
- Show smart plain-English recommendation and AI insights
- Save every prediction to user's history automatically

### 📋 IPO Listings
- Browse 560+ IPO records
- Status tabs: All / Upcoming / Open Now / Closed / Listed
- Search by IPO name
- Filter by Sector
- Filter by Listing Gain Category
- Filter by Success Probability (High / Medium / Low)
- Sort by Highest Probability / Highest Subscription / Newest / Price
- View full IPO details in a modal popup

### 📊 Analytics Dashboard
- 12 interactive Chart.js charts
- IPO Success vs Failure donut chart
- Sector-wise success and subscription performance
- Most subscribed IPOs horizontal bar chart
- Listing Gain Category distribution
- Risk and Confidence distribution pie charts
- Recent IPO trend line chart

### 🏠 Home Page
- "What is an IPO?" educational section
- How the prediction system works (4-step guide)
- Key factors affecting IPO success
- Top Recommended IPOs from database
- Most Subscribed IPOs
- Recent IPO Activity table

### 🕘 Prediction History
- All predictions saved per user account
- View IPO name, prediction, probability, risk, confidence
- Delete individual history entries
- Re-predict any past IPO with one click

### 📖 About Page
- Website purpose and features explained in plain English
- No technical jargon — suitable for all users
- Step-by-step usage guide

### 🔐 Authentication
- Signup with Name, Username, Email, Password
- Login and Logout
- Session-based user management
- SHA-256 password hashing

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, Flask |
| ML / Data | scikit-learn, pandas, NumPy |
| Database | SQLite (via Python sqlite3) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js v4 |
| Fonts | Google Fonts – Inter |
| Model Storage | Pickle (.pkl file) |

---

## 📁 Folder Structure

```
IPO-Success-Analyzer/
├── app.py                    # Main Flask application + all routes
├── model_training.py         # Model training script
├── dataset.csv               # Historical IPO dataset (5000 records)
├── ipo_model.pkl             # Trained Logistic Regression model (pickle)
├── users.db                  # SQLite database (auto-created on first run)
├── README.md                 # This file
│
├── templates/
│   ├── navbar.html           # Shared navigation bar (Jinja include)
│   ├── index.html            # Home page
│   ├── predictor.html        # IPO Prediction page
│   ├── listings.html         # IPO Listings browser
│   ├── dashboard.html        # Analytics Dashboard
│   ├── about.html            # About page
│   ├── history.html          # User prediction history
│   ├── login.html            # Login page
│   └── signup.html           # Sign Up page
│
└── static/
    ├── style.css             # Global dark theme CSS
    ├── script.js             # Predictor page JS (prediction, charts, search)
    ├── listings.js           # Listings page JS (fetch, filter, tabs, modal)
    ├── dashboard.js          # Dashboard page JS (Chart.js rendering)
    └── history.js            # History page JS (fetch, delete, re-predict)
```

---

## 📊 Dataset Explanation

**File:** `dataset.csv`  
**Records:** ~5,000 historical IPOs from Indian stock markets

### Columns

| Column | Description |
|---|---|
| `Issue Size` | Total amount raised by the IPO (in ₹ Crore) |
| `Offer Price` | Price per share offered to investors (₹) |
| `QIB` | Qualified Institutional Buyers subscription (times oversubscribed) |
| `HNI` | High Net-worth Individuals subscription (times oversubscribed) |
| `RII` | Retail Individual Investors subscription (times oversubscribed) |
| `Total` | Overall subscription across all categories |
| `IPO_Success` | Target variable: 1 = Listed at gain, 0 = Listed below issue price |

**Source:** Collected from publicly available BSE/NSE historical data.  
**Preprocessing:** Missing values dropped, features standardized using `StandardScaler`.

---

## 🤖 Final Model Features

The model uses exactly **6 features** as inputs:

```
1. Issue_Size   — Size of the IPO in ₹ Crore
2. Offer_Price  — Issue price per share
3. QIB          — Institutional investor demand
4. HNI          — High net-worth investor demand
5. RII          — Retail investor demand
6. Total        — Total subscription multiplier
```

These 6 features were selected because they represent **investor demand at all three investor categories**, which is the strongest signal for listing performance.

---

## 🧠 Why Logistic Regression?

Logistic Regression was chosen over complex models (Random Forest, SVM, Neural Networks) for the following reasons:

| Reason | Explanation |
|---|---|
| **Binary output** | IPO listing follows a binary outcome — Success (1) or Fail (0) |
| **Probability output** | Logistic Regression naturally outputs probabilities (0–1), which we use for confidence scoring |
| **Interpretability** | Coefficients directly show which features matter most |
| **Fast inference** | Instant predictions — no latency for users |
| **No overfitting** | With 5,000 records and 6 features, simpler models generalize better |
| **Accuracy** | Achieved **84.07% accuracy** and **0.91 F1-score** on the test set |

---

## 🔢 Prediction Logic

```python
# Input features
X = [Issue_Size, Offer_Price, QIB, HNI, RII, Total]

# Scale
X_scaled = StandardScaler().transform(X)

# Predict
prediction    = model.predict(X_scaled)         # 0 or 1
probabilities = model.predict_proba(X_scaled)   # [p_fail, p_success]
```

The raw model output is:
- `prediction = 1` → IPO predicted to list at a gain
- `prediction = 0` → IPO predicted to list below issue price

---

## 📐 Probability Calculation

After getting `success_prob` from the model:

| Metric | Calculation |
|---|---|
| **IPO Score** | `success_prob × 100` (0–100 scale) |
| **Risk Level** | Low if prob ≥ 0.75, High if prob < 0.45, else Medium |
| **Confidence** | Very High if prob ≥ 0.85 or < 0.15, High if ≥ 0.70 or < 0.30, else Medium / Low |
| **Recommendation** | Strong Buy / Buy / Hold / Avoid / Strong Avoid based on probability range |

**Smart Insights** are generated from subscription values:
- QIB > 10× → "Strong institutional interest — bullish signal"
- Total > 30× → "Highly oversubscribed — strong listing gain likely"
- RII < 1× → "Retail demand weak — possible listing at discount"
- HNI > 50× → "HNI oversubscription signals short-term listing pop"

---

## 👤 User Workflow

```
1. Sign Up → Create account (Name, Username, Email, Password)
2. Log In  → Authenticate with username + password
3. Predict → Search IPO name → auto-fill data → click "Predict IPO Success"
4. Results → See prediction, probability bars, donut chart, score, risk, insights
5. History → All predictions automatically saved; re-predict or delete anytime
6. Explore → Browse Listings (search, filter, sort) and Dashboard (12 charts)
7. Log Out → Session cleared
```

---

## 📈 Dashboard Explanation

The Dashboard (`/dashboard`) renders **12 Chart.js charts** using live data from `users.db`:

| # | Chart | Type | Data Source |
|---|---|---|---|
| 1 | IPO Success vs Failure | Doughnut | All IPO records |
| 2 | Sector-wise IPO Success | Grouped Bar | Per-sector aggregation |
| 3 | Avg QIB vs HNI vs RII | Bar | Overall averages |
| 4 | Most Subscribed IPOs | Horizontal Bar | Top 10 by total_sub |
| 5 | Highest Probability IPOs | Horizontal Bar | Top 10 by success_prob |
| 6 | Listing Gain Distribution | Doughnut | listing_gain_cat |
| 7 | Listing Gain by Sector | Bar | Sector avg gain proxy |
| 8 | Risk Level Distribution | Pie | IPO score-based risk |
| 9 | Confidence Distribution | Pie | success_prob ranges |
| 10 | Recent IPO Trend | Line | Last 20 IPOs |
| 11 | Sector Avg Subscription | Bar | Per-sector avg total_sub |
| 12 | Sector Success Rate | Line | Per-sector success rate % |

---

## ⚙️ Installation Steps

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ipo-success-predictor.git
cd IPO-Success-Analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask pandas scikit-learn numpy
```

### 4. Verify project structure

Ensure these files are present:
- `app.py`
- `dataset.csv`
- `ipo_model.pkl`
- `templates/` folder with all HTML files
- `static/` folder with `style.css`, `script.js`, `listings.js`, `dashboard.js`, `history.js`

---

## ▶️ How to Run Locally

```bash
# From the project root
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

The SQLite database (`users.db`) is created automatically on first run with all required tables and the IPO listings data seeded from `dataset.csv`.

> **Note:** The app runs in development mode with `debug=True`. For production, use a WSGI server like Gunicorn.

---

## 🔮 Future Scope

| Enhancement | Description |
|---|---|
| **Live IPO Data** | Integrate BSE/NSE APIs for real subscription data during IPO open period |
| **Sentiment Analysis** | Scrape and analyze news/social media sentiment for each IPO |
| **Portfolio Tracker** | Let users track applied IPOs and eventual listing profits/losses |
| **Email Alerts** | Notify users when an IPO they saved is about to open or close |
| **Advanced Models** | Experiment with XGBoost, LightGBM, or Ensemble models for higher accuracy |
| **Mobile App** | Build a React Native or Flutter version of the app |
| **Grey Market Premium** | Include GMP (Grey Market Premium) as an additional prediction feature |
| **Multi-exchange Support** | Extend beyond NSE/BSE to cover SME IPOs and international markets |

---

