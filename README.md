# ModellingDengueProject

# SOFE 4820U — Dengue Fever Monte Carlo Simulation

**Authors:** Riya Rajesh (100869701) · Nancy Emanuel (100657804)
**Course:** SOFE 4820U Modelling and Simulation · Winter 2026
**Instructor:** Dr. Anwar Abdalbari

---

## Setup

Install dependencies:

```
pip install streamlit pandas numpy scipy plotly requests
pip install statsmodels
```

---

## Run

```
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Data

Live data is pulled from the InfoDengue API automatically when the app runs (requires internet).
If the API is unreachable, the app falls back to synthetic data matching InfoDengue's schema.

Test the API manually in your browser:

```
https://info.dengue.mat.br/api/alertcity?geocode=3304557&disease=dengue&format=csv&ew_start=1&ew_end=52&ey_start=2023&ey_end=2023
```

---

## Files

```
app.py       — full application (backend + Streamlit UI)
README.md    — this file
```
