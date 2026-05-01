"""
app.py  –  IPO Success Predictor (Full Upgrade)
------------------------------------------------
Routes:
  GET  /                    → home page
  GET  /predictor           → prediction page (login required)
  POST /predict             → ML prediction API
  GET  /trends              → IPO trends / timeline page
  GET  /trends/data         → JSON data for trend charts
  GET  /dashboard           → Chart.js dashboard
  GET  /dashboard/data      → JSON aggregated dashboard data
  GET  /listings            → IPO Listings page
  GET  /listings/data       → JSON listing data (filterable)
  GET  /listings/<int:id>   → Single IPO detail JSON
  GET  /history             → Prediction history (login required)
  GET  /history/data        → JSON history for logged-in user
  DELETE /history/<int:id>  → Delete history entry
  GET  /about               → about / information page
  GET  /login               → login page
  POST /login               → process login
  GET  /signup              → signup page
  POST /signup              → process signup
  GET  /logout              → logout user
  GET  /ipo/lookup          → look up IPO details by name (AJAX)
"""

import pickle, sqlite3, hashlib, os, json, random, math

# ── Base directory (same folder as app.py) ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from functools import wraps
from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, session, flash)

app = Flask(__name__)
app.secret_key = "ipo_secret_key_2024_ultra"

# ── Load ML model ─────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "ipo_model.pkl"), "rb") as f:
    payload = pickle.load(f)

model    = payload["model"]
scaler   = payload["scaler"]
features = payload["features"]   # ["Issue_Size","Offer_Price","QIB","HNI","RII","Total"]

# ── Synthetic IPO data ────────────────────────────────────────────────────────
SECTORS = [
    "Technology", "Finance", "Healthcare", "Manufacturing",
    "Retail & Consumer", "Energy", "Infrastructure",
    "Pharma & Biotech", "Real Estate", "Telecom"
]

LISTING_GAIN_CATS = ["Exceptional (>50%)", "Strong (20-50%)", "Moderate (5-20%)",
                     "Flat (-5% to 5%)", "Loss (< -5%)"]

IPO_NAME_PREFIXES = [
    "Apex", "Nova", "Zeta", "Prime", "Vega", "Atlas", "Crest", "Nexus",
    "Allied", "Optima", "Horizon", "Pinnacle", "Summit", "Pioneer", "Stellar",
    "Quantum", "Fusion", "Vector", "Matrix", "Origin", "Prism", "Zenith",
    "Metro", "Global", "National", "Bluechip", "Capital", "Tech", "Smart",
    "Bright", "Clear", "Swift", "Micro", "Macro", "Dynamic", "Rapid",
    "Shield", "Armor", "Crown", "Diamond", "Emerald", "Sapphire", "Gold",
    "Silver", "Platinum", "Ruby", "Pearl", "Crystal", "Vision"
]

IPO_NAME_SUFFIXES = [
    "Technologies", "Solutions", "Innovations", "Systems", "Enterprises",
    "Holdings", "Industries", "Finance", "Capital", "Ventures",
    "Corp", "Limited", "Group", "Services", "Labs",
    "Networks", "Digital", "Dynamics", "Partners", "Associates"
]

# ── SQLite helpers ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # Users table (existing)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT,
                username TEXT    UNIQUE NOT NULL,
                email    TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL
            )
        """)
        # Try to add 'name' column if it doesn't exist (migration)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
        except Exception:
            pass

        # IPO listings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ipo_listings (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT NOT NULL,
                sector            TEXT,
                issue_size        REAL,
                offer_price       REAL,
                qib               REAL,
                hni               REAL,
                rii               REAL,
                total_sub         REAL,
                past_performance  REAL,
                sector_performance REAL,
                listing_gain_cat  TEXT,
                listing_date      TEXT,
                open_date         TEXT,
                close_date        TEXT,
                allotment_date    TEXT,
                predicted_success INTEGER,
                success_prob      REAL,
                ipo_success       INTEGER
            )
        """)

        # Prediction history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL,
                ipo_name         TEXT,
                offer_price      REAL,
                issue_size       REAL,
                qib              REAL,
                hni              REAL,
                rii              REAL,
                total_sub        REAL,
                prediction       INTEGER,
                success_prob     REAL,
                failure_prob     REAL,
                confidence       TEXT,
                risk_level       TEXT,
                recommendation   TEXT,
                predicted_at     TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()

        # Seed IPO listings if empty
        count = conn.execute("SELECT COUNT(*) FROM ipo_listings").fetchone()[0]
        if count == 0:
            _seed_ipo_listings(conn)

def _seed_ipo_listings(conn):
    """Seed the ipo_listings table with synthetic data derived from the real dataset."""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
        df.columns = [c.strip() for c in df.columns]
        # Map columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            col_map[c] = cl
        df.rename(columns=col_map, inplace=True)

        # Normalize column names
        if "issue_size" not in df.columns and "issue size" in [c.lower() for c in df.columns]:
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        df = df.dropna(subset=["issue_size"], how="all") if "issue_size" in df.columns else df
        df = df.fillna(0)

        random.seed(42)
        used_names = set()
        base_date = datetime(2020, 1, 15)

        rows = []
        for i, row in df.iterrows():
            # Generate unique name
            while True:
                name = f"{random.choice(IPO_NAME_PREFIXES)} {random.choice(IPO_NAME_SUFFIXES)}"
                if name not in used_names:
                    used_names.add(name)
                    break

            sector = random.choice(SECTORS)
            issue_size   = float(row.get("issue_size",   row.get("issue size", 1000)))
            offer_price  = float(row.get("offer_price",  row.get("offer price", 300)))
            qib          = float(row.get("qib",  0))
            hni          = float(row.get("hni",  0))
            rii          = float(row.get("rii",  0))
            total_sub    = float(row.get("total", 0))
            ipo_success  = int(row.get("ipo_success", row.get("ipo success", 0)))

            # Past/sector performance as a proxy from dataset patterns
            past_perf   = round(min(1.0, total_sub / 200.0), 2)
            sector_perf = round(random.uniform(0.3, 0.95), 2)

            # Listing gain category derived from total subscription
            if total_sub >= 50:
                lgc = "Exceptional (>50%)"
            elif total_sub >= 20:
                lgc = "Strong (20-50%)"
            elif total_sub >= 5:
                lgc = "Moderate (5-20%)"
            elif total_sub >= 0:
                lgc = "Flat (-5% to 5%)"
            else:
                lgc = "Loss (< -5%)"

            # Dates (spread across 2018–2024)
            offset_days = i * 4 + random.randint(-2, 10)
            open_date      = (datetime(2018, 1, 1) + timedelta(days=offset_days)).strftime("%Y-%m-%d")
            close_date     = (datetime(2018, 1, 1) + timedelta(days=offset_days + 3)).strftime("%Y-%m-%d")
            allotment_date = (datetime(2018, 1, 1) + timedelta(days=offset_days + 7)).strftime("%Y-%m-%d")
            listing_date   = (datetime(2018, 1, 1) + timedelta(days=offset_days + 10)).strftime("%Y-%m-%d")

            # Model prediction
            try:
                X = pd.DataFrame([[issue_size, offer_price, qib, hni, rii, total_sub]],
                                  columns=features)
                X_sc = scaler.transform(X)
                pred      = int(model.predict(X_sc)[0])
                prob      = float(model.predict_proba(X_sc)[0][1])
            except Exception:
                pred = ipo_success
                prob = 0.75 if ipo_success else 0.25

            rows.append((name, sector, issue_size, offer_price, qib, hni, rii,
                         total_sub, past_perf, sector_perf, lgc,
                         listing_date, open_date, close_date, allotment_date,
                         pred, round(prob, 4), ipo_success))

        conn.executemany("""
            INSERT INTO ipo_listings
              (name, sector, issue_size, offer_price, qib, hni, rii, total_sub,
               past_performance, sector_performance, listing_gain_cat,
               listing_date, open_date, close_date, allotment_date,
               predicted_success, success_prob, ipo_success)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        print(f"[DB] Seeded {len(rows)} IPO listings.")
    except Exception as e:
        print(f"[DB] Seed error: {e}")

init_db()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Prediction helpers ────────────────────────────────────────────────────────
def calc_confidence(prob):
    if prob >= 0.85 or prob <= 0.15:
        return "Very High"
    if prob >= 0.70 or prob <= 0.30:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Medium"
    return "Low"

def calc_risk(prob):
    if prob >= 0.70:
        return "Low"
    if prob >= 0.45:
        return "Medium"
    return "High"

def calc_recommendation(prob, is_success):
    if is_success:
        if prob >= 0.85:
            return "Strong Buy — Exceptional listing gain expected"
        if prob >= 0.70:
            return "Buy — Good listing gain likely"
        return "Cautious Buy — Moderate upside, watch market"
    else:
        if prob <= 0.20:
            return "Avoid — High risk of listing below offer price"
        return "Skip — Weak indicators, better opportunities likely"

def generate_insights(qib, hni, rii, total_sub, offer_price, is_success, prob):
    insights = []
    if qib >= 10:
        insights.append({"icon": "📊", "type": "positive",
                          "text": f"Strong institutional confidence: QIB {qib}× shows high interest from mutual funds & banks."})
    elif qib >= 2:
        insights.append({"icon": "📊", "type": "neutral",
                          "text": f"Moderate QIB subscription ({qib}×) — institutional interest present but not exceptional."})
    else:
        insights.append({"icon": "⚠️", "type": "negative",
                          "text": f"Low QIB subscription ({qib}×) — limited institutional confidence."})

    if hni >= 20:
        insights.append({"icon": "💼", "type": "neutral",
                          "text": f"Very high HNI ({hni}×) — may indicate speculative activity."})
    elif hni >= 5:
        insights.append({"icon": "💼", "type": "positive",
                          "text": f"Solid HNI participation ({hni}×) — high net-worth investors are engaged."})

    if rii >= 5:
        insights.append({"icon": "👥", "type": "positive",
                          "text": f"Strong retail demand: RII {rii}× shows high public interest."})
    elif rii < 1:
        insights.append({"icon": "⚠️", "type": "negative",
                          "text": f"Weak retail participation ({rii}×) — low public demand detected."})

    if total_sub >= 30:
        insights.append({"icon": "🚀", "type": "positive",
                          "text": f"Exceptional overall demand: {total_sub}× total subscription is a very strong signal."})
    elif total_sub >= 10:
        insights.append({"icon": "📈", "type": "positive",
                          "text": f"Good total demand: {total_sub}× subscription — positive market reception."})
    elif total_sub < 2:
        insights.append({"icon": "⚠️", "type": "negative",
                          "text": f"Undersubscribed ({total_sub}×) — demand below expectations, high listing risk."})

    if offer_price > 800:
        insights.append({"icon": "💰", "type": "neutral",
                          "text": f"High offer price (₹{offer_price}) may limit retail participation."})

    if is_success:
        insights.append({"icon": "✅", "type": "positive",
                          "text": "Overall: Subscription figures suggest likely listing at a premium."})
    else:
        insights.append({"icon": "❌", "type": "negative",
                          "text": "Overall: Weak subscription data — listing below offer price is likely."})
    return insights

# ── Page Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def home():
    with get_db() as conn:
        total_ipos   = conn.execute("SELECT COUNT(*) FROM ipo_listings").fetchone()[0]
        success_ipos = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE ipo_success=1").fetchone()[0]
        avg_prob     = conn.execute("SELECT AVG(success_prob) FROM ipo_listings").fetchone()[0] or 0

        # Best sector
        best_sector_row = conn.execute("""
            SELECT sector, COUNT(*) as cnt, SUM(ipo_success) as sc
            FROM ipo_listings GROUP BY sector ORDER BY (CAST(sc AS FLOAT)/cnt) DESC LIMIT 1
        """).fetchone()
        best_sector = best_sector_row["sector"] if best_sector_row else "Technology"

        # Highest listing gain cat
        top_lgc_row = conn.execute("""
            SELECT listing_gain_cat, COUNT(*) as cnt FROM ipo_listings
            WHERE ipo_success=1 GROUP BY listing_gain_cat ORDER BY cnt DESC LIMIT 1
        """).fetchone()
        top_lgc = top_lgc_row["listing_gain_cat"] if top_lgc_row else "Strong (20-50%)"

        # Top recommended
        top_ipos = conn.execute("""
            SELECT * FROM ipo_listings WHERE predicted_success=1
            ORDER BY success_prob DESC LIMIT 6
        """).fetchall()

        # Most subscribed
        most_subscribed = conn.execute("""
            SELECT * FROM ipo_listings ORDER BY total_sub DESC LIMIT 6
        """).fetchall()

        # Recent IPO activity (last 8 by listing date)
        recent_ipos = conn.execute("""
            SELECT * FROM ipo_listings ORDER BY listing_date DESC LIMIT 8
        """).fetchall()

    stats = {
        "total_ipos":    total_ipos,
        "success_rate":  round((success_ipos / total_ipos * 100) if total_ipos else 0, 1),
        "avg_prob":      round(avg_prob * 100, 1),
        "best_sector":   best_sector,
        "top_lgc":       top_lgc,
    }
    return render_template("index.html",
                           user=session.get("username"),
                           stats=stats,
                           top_ipos=[dict(r) for r in top_ipos],
                           most_subscribed=[dict(r) for r in most_subscribed],
                           recent_ipos=[dict(r) for r in recent_ipos])

@app.route("/predictor")
@login_required
def predictor():
    return render_template("predictor.html", user=session.get("username"))

@app.route("/trends")
def trends():
    return render_template("trends.html", user=session.get("username"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", user=session.get("username"))

@app.route("/about")
def about():
    return render_template("about.html", user=session.get("username"))

@app.route("/listings")
def listings():
    return render_template("listings.html", user=session.get("username"))

@app.route("/history")
@login_required
def history():
    return render_template("history.html", user=session.get("username"))

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("predictor"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? AND password = ?",
                (username, hash_password(password))
            ).fetchone()
        if user:
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["name"]     = user["name"] or user["username"]
            flash(f"Welcome back, {user['username']}! 👋", "success")
            return redirect(url_for("predictor"))
        else:
            flash("Incorrect username or password.", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("predictor"))
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not all([username, email, password, confirm]):
            flash("Please fill in all fields.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users (name, username, email, password) VALUES (?, ?, ?, ?)",
                        (name, username, email, hash_password(password))
                    )
                    conn.commit()
                flash("Account created successfully! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username or email already exists.", "error")
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        input_values = []
        for feat in features:
            val = data.get(feat)
            if val is None:
                return jsonify({"error": f"Missing field: {feat}"}), 400
            input_values.append(float(val))

        X        = pd.DataFrame([input_values], columns=features)
        X_scaled = scaler.transform(X)
        prediction    = int(model.predict(X_scaled)[0])
        probas        = model.predict_proba(X_scaled)[0]
        success_prob  = float(probas[1])
        failure_prob  = float(probas[0])

        is_success   = prediction == 1
        confidence   = calc_confidence(success_prob)
        risk_level   = calc_risk(success_prob)
        recommendation = calc_recommendation(success_prob, is_success)

        qib        = float(data.get("QIB", 0))
        hni        = float(data.get("HNI", 0))
        rii        = float(data.get("RII", 0))
        total_sub  = float(data.get("Total", 0))
        offer_price = float(data.get("Offer_Price", 0))

        insights = generate_insights(qib, hni, rii, total_sub, offer_price, is_success, success_prob)

        # Save to prediction history if user is logged in
        if "user_id" in session:
            ipo_name = data.get("ipo_name", "Manual Entry")
            try:
                with get_db() as conn:
                    conn.execute("""
                        INSERT INTO prediction_history
                          (user_id, ipo_name, offer_price, issue_size, qib, hni, rii,
                           total_sub, prediction, success_prob, failure_prob,
                           confidence, risk_level, recommendation)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        session["user_id"], ipo_name,
                        float(data.get("Offer_Price", 0)),
                        float(data.get("Issue_Size", 0)),
                        qib, hni, rii, total_sub,
                        prediction,
                        round(success_prob, 4),
                        round(failure_prob, 4),
                        confidence, risk_level, recommendation
                    ))
                    conn.commit()
            except Exception as e:
                print(f"History save error: {e}")

        return jsonify({
            "prediction":          prediction,
            "success_probability": round(success_prob, 4),
            "failure_probability": round(failure_prob, 4),
            "confidence":          confidence,
            "risk_level":          risk_level,
            "recommendation":      recommendation,
            "insights":            insights,
        })
    except ValueError as ve:
        return jsonify({"error": f"Invalid value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ipo/lookup")
def ipo_lookup():
    """Look up IPO details by name for auto-fill."""
    name = request.args.get("name", "").strip()
    if not name or len(name) < 2:
        return jsonify([])
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM ipo_listings
            WHERE name LIKE ? ORDER BY name LIMIT 10
        """, (f"%{name}%",)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/trends/data")
def trends_data():
    """Return aggregated chart data from the dataset."""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
        df.columns = [c.strip() for c in df.columns]
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        df.dropna(subset=["qib","hni","rii","total"], how="all", inplace=True)
        df.fillna(0, inplace=True)

        df["Period"] = pd.cut(df.index, bins=10, labels=[f"P{i}" for i in range(1, 11)])
        sr = df.groupby("Period", observed=True)["ipo_success"].mean().mul(100).round(1)
        ts = df.groupby("Period", observed=True)["total"].mean().round(2)

        # Sector-wise from db
        with get_db() as conn:
            sector_rows = conn.execute("""
                SELECT sector,
                       COUNT(*) as total,
                       SUM(ipo_success) as success,
                       AVG(total_sub) as avg_sub
                FROM ipo_listings
                GROUP BY sector
                ORDER BY success DESC
            """).fetchall()
            timeline_ipos = conn.execute("""
                SELECT name, sector, offer_price, listing_date, open_date,
                       close_date, allotment_date, listing_gain_cat, success_prob,
                       predicted_success
                FROM ipo_listings
                ORDER BY listing_date DESC LIMIT 12
            """).fetchall()

        sector_data = {
            "labels":  [r["sector"] for r in sector_rows],
            "success": [r["success"] for r in sector_rows],
            "total":   [r["total"] for r in sector_rows],
            "avg_sub": [round(r["avg_sub"] or 0, 2) for r in sector_rows],
        }

        return jsonify({
            "periods":       sr.index.tolist(),
            "success_rate":  sr.values.tolist(),
            "avg_total_sub": ts.values.tolist(),
            "success_count": int(df["ipo_success"].sum()),
            "fail_count":    int((df["ipo_success"] == 0).sum()),
            "avg_qib":   round(float(df["qib"].mean()),   2),
            "avg_hni":   round(float(df["hni"].mean()),   2),
            "avg_rii":   round(float(df["rii"].mean()),   2),
            "avg_total": round(float(df["total"].mean()), 2),
            "sector":    sector_data,
            "timeline_ipos": [dict(r) for r in timeline_ipos],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard/data")
def dashboard_data():
    """Return all chart data for the dashboard."""
    try:
        with get_db() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM ipo_listings").fetchone()[0]
            success_c  = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE ipo_success=1").fetchone()[0]
            fail_c     = total - success_c
            avg_prob   = conn.execute("SELECT AVG(success_prob) FROM ipo_listings").fetchone()[0] or 0

            # Best sector
            best_sec = conn.execute("""
                SELECT sector, CAST(SUM(ipo_success) AS FLOAT)/COUNT(*) as rate
                FROM ipo_listings GROUP BY sector ORDER BY rate DESC LIMIT 1
            """).fetchone()

            # Top listing gain cat
            top_lgc = conn.execute("""
                SELECT listing_gain_cat, COUNT(*) as cnt
                FROM ipo_listings WHERE ipo_success=1
                GROUP BY listing_gain_cat ORDER BY cnt DESC LIMIT 1
            """).fetchone()

            # Sector success bar
            sec_rows = conn.execute("""
                SELECT sector, SUM(ipo_success) as s, COUNT(*) as t,
                       AVG(total_sub) as avg_sub, AVG(success_prob) as avg_prob
                FROM ipo_listings GROUP BY sector
            """).fetchall()

            # Subscription averages
            sub_avg = conn.execute("""
                SELECT AVG(qib) aq, AVG(hni) ah, AVG(rii) ar, AVG(total_sub) at
                FROM ipo_listings
            """).fetchone()

            # Most subscribed IPOs (top 8)
            most_sub = conn.execute("""
                SELECT name, total_sub, success_prob, ipo_success
                FROM ipo_listings ORDER BY total_sub DESC LIMIT 8
            """).fetchall()

            # Highest probability
            high_prob = conn.execute("""
                SELECT name, success_prob, sector
                FROM ipo_listings ORDER BY success_prob DESC LIMIT 8
            """).fetchall()

            # Listing gain cat distribution
            lgc_dist = conn.execute("""
                SELECT listing_gain_cat, COUNT(*) as cnt
                FROM ipo_listings GROUP BY listing_gain_cat
            """).fetchall()

            # Avg listing gain by sector (proxy: avg success_prob)
            lgc_sec = conn.execute("""
                SELECT sector, AVG(success_prob)*100 as avg_gain_proxy
                FROM ipo_listings GROUP BY sector
            """).fetchall()

            # Risk distribution
            risk_rows = conn.execute("""
                SELECT
                  SUM(CASE WHEN success_prob >= 0.70 THEN 1 ELSE 0 END) as low_risk,
                  SUM(CASE WHEN success_prob >= 0.45 AND success_prob < 0.70 THEN 1 ELSE 0 END) as med_risk,
                  SUM(CASE WHEN success_prob < 0.45 THEN 1 ELSE 0 END) as high_risk
                FROM ipo_listings
            """).fetchone()

            # Confidence distribution
            conf_rows = conn.execute("""
                SELECT
                  SUM(CASE WHEN success_prob >= 0.85 OR success_prob <= 0.15 THEN 1 ELSE 0 END) as very_high,
                  SUM(CASE WHEN (success_prob >= 0.70 AND success_prob < 0.85) OR (success_prob > 0.15 AND success_prob <= 0.30) THEN 1 ELSE 0 END) as high,
                  SUM(CASE WHEN (success_prob >= 0.60 AND success_prob < 0.70) OR (success_prob > 0.30 AND success_prob <= 0.40) THEN 1 ELSE 0 END) as medium,
                  SUM(CASE WHEN success_prob > 0.40 AND success_prob < 0.60 THEN 1 ELSE 0 END) as low
                FROM ipo_listings
            """).fetchone()

            # Recent trend (by listing_date periods)
            recent_trend = conn.execute("""
                SELECT substr(listing_date,1,7) as month,
                       COUNT(*) as cnt,
                       AVG(success_prob)*100 as avg_prob
                FROM ipo_listings
                GROUP BY month ORDER BY month DESC LIMIT 12
            """).fetchall()

        # Feature importance from model coefficients
        coefs = model.coef_[0].tolist()
        feature_importance = [{"name": f, "coef": round(c, 4)}
                               for f, c in zip(features, coefs)]
        feature_importance.sort(key=lambda x: abs(x["coef"]), reverse=True)

        return jsonify({
            "summary": {
                "total": total,
                "success": success_c,
                "fail": fail_c,
                "avg_prob": round(avg_prob * 100, 1),
                "best_sector": best_sec["sector"] if best_sec else "—",
                "top_lgc": top_lgc["listing_gain_cat"] if top_lgc else "—",
            },
            "sector_bar": {
                "labels":   [r["sector"] for r in sec_rows],
                "success":  [r["s"] for r in sec_rows],
                "total":    [r["t"] for r in sec_rows],
                "avg_sub":  [round(r["avg_sub"] or 0, 2) for r in sec_rows],
                "avg_prob": [round((r["avg_prob"] or 0)*100, 1) for r in sec_rows],
            },
            "subscription_avg": {
                "qib":   round(sub_avg["aq"] or 0, 2),
                "hni":   round(sub_avg["ah"] or 0, 2),
                "rii":   round(sub_avg["ar"] or 0, 2),
                "total": round(sub_avg["at"] or 0, 2),
            },
            "most_subscribed": [dict(r) for r in most_sub],
            "highest_prob":    [dict(r) for r in high_prob],
            "lgc_distribution": {
                "labels": [r["listing_gain_cat"] for r in lgc_dist],
                "counts": [r["cnt"] for r in lgc_dist],
            },
            "lgc_by_sector": {
                "labels": [r["sector"] for r in lgc_sec],
                "values": [round(r["avg_gain_proxy"] or 0, 1) for r in lgc_sec],
            },
            "risk_distribution": {
                "low":    risk_rows["low_risk"] or 0,
                "medium": risk_rows["med_risk"] or 0,
                "high":   risk_rows["high_risk"] or 0,
            },
            "confidence_distribution": {
                "very_high": conf_rows["very_high"] or 0,
                "high":      conf_rows["high"] or 0,
                "medium":    conf_rows["medium"] or 0,
                "low":       conf_rows["low"] or 0,
            },
            "recent_trend": [dict(r) for r in reversed(list(recent_trend))],
            "feature_importance": feature_importance,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/listings/data")
def listings_data():
    """Return filtered/sorted IPO listings."""
    try:
        search   = request.args.get("search", "").strip()
        sector   = request.args.get("sector", "").strip()
        lgc      = request.args.get("lgc", "").strip()
        status   = request.args.get("status", "").strip()
        prob     = request.args.get("prob", "").strip()   # high / medium / low
        sort_by  = request.args.get("sort", "prob_desc")
        page     = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 12))

        today = datetime.now().strftime("%Y-%m-%d")

        STATUS_SQL = {
            "upcoming": ("open_date > ?",                        [today]),
            "open":     ("open_date <= ? AND close_date >= ?",   [today, today]),
            "closed":   ("close_date < ? AND listing_date > ?",  [today, today]),
            "listed":   ("listing_date <= ?",                    [today]),
        }

        conditions = []
        params = []
        if search:
            conditions.append("name LIKE ?")
            params.append(f"%{search}%")
        if sector:
            conditions.append("sector = ?")
            params.append(sector)
        if lgc:
            conditions.append("listing_gain_cat = ?")
            params.append(lgc)
        if prob == "high":
            conditions.append("success_prob >= 0.7")
        elif prob == "medium":
            conditions.append("success_prob >= 0.45 AND success_prob < 0.7")
        elif prob == "low":
            conditions.append("success_prob < 0.45")
        if status and status in STATUS_SQL:
            sql_frag, sql_params = STATUS_SQL[status]
            conditions.append(sql_frag)
            params.extend(sql_params)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        order_map = {
            "prob_desc":  "success_prob DESC",
            "prob_asc":   "success_prob ASC",
            "sub_desc":   "total_sub DESC",
            "newest":     "listing_date DESC",
            "oldest":     "listing_date ASC",
            "price_desc": "offer_price DESC",
            "price_asc":  "offer_price ASC",
        }
        order = order_map.get(sort_by, "success_prob DESC")

        with get_db() as conn:
            total_rows = conn.execute(
                f"SELECT COUNT(*) FROM ipo_listings {where}", params
            ).fetchone()[0]
            offset = (page - 1) * per_page
            rows = conn.execute(
                f"SELECT * FROM ipo_listings {where} ORDER BY {order} LIMIT ? OFFSET ?",
                params + [per_page, offset]
            ).fetchall()
            sectors_list = conn.execute(
                "SELECT DISTINCT sector FROM ipo_listings ORDER BY sector"
            ).fetchall()
            lgc_list = conn.execute(
                "SELECT DISTINCT listing_gain_cat FROM ipo_listings ORDER BY listing_gain_cat"
            ).fetchall()
            # Tab counts
            cnt_all      = conn.execute("SELECT COUNT(*) FROM ipo_listings").fetchone()[0]
            cnt_upcoming = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE open_date > ?", (today,)).fetchone()[0]
            cnt_open     = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE open_date <= ? AND close_date >= ?", (today, today)).fetchone()[0]
            cnt_closed   = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE close_date < ? AND listing_date > ?", (today, today)).fetchone()[0]
            cnt_listed   = conn.execute("SELECT COUNT(*) FROM ipo_listings WHERE listing_date <= ?", (today,)).fetchone()[0]

        return jsonify({
            "total":    total_rows,
            "page":     page,
            "per_page": per_page,
            "pages":    math.ceil(max(total_rows, 1) / per_page),
            "items":    [dict(r) for r in rows],
            "sectors":  [r["sector"] for r in sectors_list],
            "lgc_opts": [r["listing_gain_cat"] for r in lgc_list],
            "status_counts": {
                "all":      cnt_all,
                "upcoming": cnt_upcoming,
                "open":     cnt_open,
                "closed":   cnt_closed,
                "listed":   cnt_listed,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/listings/<int:ipo_id>")
def listing_detail(ipo_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM ipo_listings WHERE id=?", (ipo_id,)).fetchone()
    if not row:
        return jsonify({"error": "IPO not found"}), 404
    return jsonify(dict(row))

@app.route("/history/data")
@login_required
def history_data():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM prediction_history
            WHERE user_id=? ORDER BY predicted_at DESC
        """, (session["user_id"],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/history/<int:entry_id>", methods=["DELETE"])
@login_required
def delete_history(entry_id):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE id=? AND user_id=?",
            (entry_id, session["user_id"])
        )
        conn.commit()
    return jsonify({"status": "deleted"})

if __name__ == "__main__":
    print("IPO Predictor (Full Upgrade) -> http://127.0.0.1:5000")
    # Disable the reloader to avoid `signal` usage from a non-main thread
    # (prevents: ValueError: signal only works in main thread of the main interpreter)
    app.run(debug=True, use_reloader=False)