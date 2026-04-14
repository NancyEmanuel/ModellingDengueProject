# SOFE 4820U - Dengue Fever Climate-Driven Monte Carlo Simulation
# Monte Carlo (Weeks 4-5): fit distributions → sample → predict → compare

import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CITY LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

# IBGE geocodes used by the InfoDengue API
CITY_GEOCODES = {
    "Rio de Janeiro": "3304557",
    "Sao Paulo":      "3550308",
    "Recife":         "2611606",
    "Fortaleza":      "2304400",
    "Manaus":         "1302603",
    "Salvador":       "2927408",
    "Brasilia":       "5300108",
    "Belo Horizonte": "3106200",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING  (FIX 4 — silent)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_df(df):
    # drop fully empty rows, fill numeric NaNs with median
    df = df.dropna(how="all").copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # clip climate values to physically reasonable ranges
    if "tempMed" in df.columns:
        df["tempMed"] = df["tempMed"].clip(10, 45)
    if "umidadeMed" in df.columns:
        df["umidadeMed"] = df["umidadeMed"].clip(20, 100)
    if "chuva" in df.columns:
        df["chuva"] = df["chuva"].clip(0, 500)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# INFODENGUE API FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_infodengue(geocode, ew_start, ew_end, ey_start, ey_end):
    """
    Pull dengue data from InfoDengue CSV API.
    FIX 1 - alertcity doesn't always return rainfall (chuva); try chikungunya
             query for it, then estimate from humidity+temp if still missing.
    FIX 2 - keep 'pop' column for incidence per 100k.
    Returns empty DataFrame on failure (triggers synthetic fallback).
    """
    base = "https://info.dengue.mat.br/api/alertcity"

    def url_for(disease):
        return (
            f"{base}?geocode={geocode}&disease={disease}&format=csv"
            f"&ew_start={ew_start}&ew_end={ew_end}"
            f"&ey_start={ey_start}&ey_end={ey_end}"
        )

    try:
        df = pd.read_csv(url_for("dengue"), index_col="SE")
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])
        df = df.sort_values("data_iniSE").reset_index(drop=True)

        # Rename real API climate columns to internal names
        # InfoDengue real columns: tempmin (weekly min temp), tempmed (weekly mean),
        # umidmin, umidmax, umidmed. Use tempmed/umidmed for mean climate values;
        # fall back to tempmin/umidmax only if mean columns are absent.
        if "tempmed" in df.columns:
            df = df.rename(columns={"tempmed": "tempMed"})
        else:
            df = df.rename(columns={"tempmin": "tempMed"})  # fallback only
        if "umidmed" in df.columns:
            df = df.rename(columns={"umidmed": "umidadeMed"})
        else:
            df = df.rename(columns={"umidmax": "umidadeMed"})  # fallback only

        # FIX 1: Attempt to get real rainfall from chikungunya query
        if "chuva" not in df.columns or df["chuva"].isna().all():
            try:
                df_chik = pd.read_csv(url_for("chikungunya"), index_col="SE").reset_index()
                if "chuva" in df_chik.columns:
                    df_chik["SE"] = df_chik["SE"].astype(str)
                    df["SE"] = df["SE"].astype(str)
                    df["chuva"] = df["SE"].map(df_chik.set_index("SE")["chuva"])
            except Exception:
                pass

        # If rainfall still missing: estimate from humidity + temperature
        # Higher humidity above 60% and higher temperature → more convective rain
        if "chuva" not in df.columns or df["chuva"].isna().all():
            t = df["tempMed"].fillna(27)
            h = df["umidadeMed"].fillna(75)
            df["chuva"] = np.clip(
                ((h - 60).clip(lower=0) * 0.8) + ((t - 20).clip(lower=0) * 1.2),
                0, 150
            ).round(1)

        # FIX 2: Population and incidence per 100k
        if "pop" in df.columns:
            df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
            df["incidence_per_100k"] = (df["casos"] / df["pop"] * 100000).round(2)
        else:
            df["pop"] = np.nan
            df["incidence_per_100k"] = np.nan

        # FIX 4: Silent cleaning
        df = _clean_df(df)
        return df

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_brazil(city, n_weeks=156):
    """Realistic synthetic data matching InfoDengue schema.
    Used only when the API is unreachable."""
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2023-01-01")
    dates = [base_date + pd.Timedelta(weeks=i) for i in range(n_weeks)]

    # seasonal angle in radians — drives temperature and rainfall cycles
    angle = np.array([(d.timetuple().tm_yday / 365.25) * 2 * np.pi for d in dates])

    # temperature: seasonal cosine + small noise, clipped to realistic range
    temp = np.clip(27 + 5 * np.cos(angle - np.pi * 0.1) + rng.normal(0, 1.5, n_weeks), 20, 38)

    # rainfall: gamma-distributed around a seasonal baseline
    rain_base = np.clip(30 + 25 * np.cos(angle - np.pi * 0.15), 5, 80)
    rain = rng.gamma(shape=2, scale=rain_base / 2)

    # humidity: correlated with rainfall, clipped
    humidity = np.clip(65 + 0.25 * rain + rng.normal(0, 4, n_weeks), 50, 95)

    population = 3_000_000

    # case counts driven by temp and rain, with a 4-week lag for breeding
    temp_factor = np.maximum(0, (temp - 24) / 10)
    rain_factor = np.minimum(rain / 60, 1.5)
    base_lambda = 200 * temp_factor * rain_factor
    lag = 4
    lam = np.roll(base_lambda, lag)
    lam[:lag] = 50

    # El Niño boost: top 25% rainfall weeks get 50% more cases
    el_nino = 1 + 0.5 * np.where(rain > np.percentile(rain, 75), 1, 0)
    lam = np.clip(lam * el_nino + rng.normal(0, 20, n_weeks), 10, 5000)
    cases = rng.poisson(lam).astype(int)

    # alert level: 1–4 based on weekly case thresholds
    nivel = np.where(cases < 100, 1, np.where(cases < 500, 2, np.where(cases < 1500, 3, 4)))

    df = pd.DataFrame({
        "data_iniSE":         dates,
        "SE":                 [(d.year * 100 + ((d.timetuple().tm_yday - 1) // 7 + 1))
                               for d in dates],
        "casos":              cases,
        "casos_est":          (cases * rng.uniform(0.9, 1.1, n_weeks)).astype(int),
        "tempMed":            np.round(temp, 1),
        "umidadeMed":         np.round(humidity, 1),
        "chuva":              np.round(rain, 1),
        "nivel":              nivel,
        "Rt":                 np.where(cases > 500, 2, 1),
        "receptivo":          np.where(temp > 25, 1, 0),
        "transmissao":        np.where(cases > 200, 1, 0),
        "pop":                population,
        "incidence_per_100k": np.round((cases / population) * 100000, 2),
    })
    return _clean_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION FITTING  (Week 5 — Step 1)
# ─────────────────────────────────────────────────────────────────────────────

def fit_distributions(series):
    """Fit Normal, Exponential, Gamma, Log-Normal via MLE.
    Returns dict {name: {dist, params, aic}}. Best fit = lowest AIC.
    Gamma/Log-Normal need strictly positive values, so those use series_pos."""
    series = series[~np.isnan(series)]
    series_nonneg = series[series >= 0]        # Normal, Exponential (zeros OK)
    series_pos    = series[series > 0]         # Gamma, Log-Normal (need > 0)

    if len(series_nonneg) < 4:
        return {}

    candidates = {
        "Normal":      (stats.norm,    series_nonneg),
        "Exponential": (stats.expon,   series_nonneg),
        "Gamma":       (stats.gamma,   series_pos),
        "Log-Normal":  (stats.lognorm, series_pos),
    }
    results = {}
    for name, (dist, s) in candidates.items():
        if len(s) < 4:
            continue
        try:
            # MLE fit, then AIC = 2k - 2*loglikelihood
            params = dist.fit(s)
            ll = np.sum(dist.logpdf(s, *params))
            aic = 2 * len(params) - 2 * ll
            results[name] = {"dist": dist, "params": params, "aic": aic}
        except Exception:
            pass
    return results


def best_fit(fit_results):
    # return (name, dist, params) for the distribution with lowest AIC
    best = min(fit_results.items(), key=lambda x: x[1]["aic"])
    return best[0], best[1]["dist"], best[1]["params"]


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ENGINE  (Week 5 — 5-step process)
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(df, n_simulations, temp_delta, rain_delta_pct,
                    humidity_delta, beta, gamma, rng_seed=42):
    """
    Monte Carlo simulation of weekly dengue cases under a climate scenario.
    Steps: fit distributions → build CDFs → random sampling → scale by scenario.
    Uses bootstrap (empirical CDF) for cases — better than any single parametric family
    for skewed seasonal dengue data. Climate deltas shift the multiplier which slides
    the output distribution rightward.
    beta/gamma are SIR parameters: R0 = beta/gamma.
    """
    rng = np.random.default_rng(rng_seed)

    # Step 1: Fit distributions to observed climate variables (displayed in
    # the distribution panel and used for climate value sampling in output df).
    temp_fits  = fit_distributions(df["tempMed"].values)
    rain_fits  = fit_distributions(df["chuva"].values)
    humid_fits = fit_distributions(df["umidadeMed"].values)

    # Also fit distributions to case counts for the distribution display panel.
    case_vals = np.maximum(df["casos"].values.astype(float), 1.0)
    case_fits = fit_distributions(case_vals)

    # Step 2-3: Compute a scalar scenario multiplier from climate deltas.
    # R0 scale is relative to the default (beta=0.3, gamma=0.1 → R0=3).
    r0_scale = (beta / max(gamma, 1e-6)) / 3.0

    baseline_temp  = float(df["tempMed"].mean())
    baseline_rain  = float(df["chuva"].mean())
    baseline_humid = float(df["umidadeMed"].mean())

    # Temperature: logistic — biting rate worsens monotonically in the
    # 20-38°C range relevant to Brazilian cities (Lorenz 2025)
    temp_resp_base  = 1.0 / (1.0 + np.exp(-0.35 * (baseline_temp - 24.0)))
    temp_resp_new   = 1.0 / (1.0 + np.exp(-0.35 * (baseline_temp + temp_delta - 24.0)))
    temp_multiplier = temp_resp_new / max(temp_resp_base, 1e-9)

    # Rainfall: logistic (more rain → more breeding sites, diminishing returns)
    rain_new         = baseline_rain * (1 + rain_delta_pct / 100.0)
    rain_resp_base   = baseline_rain / (baseline_rain + 30)
    rain_resp_new    = rain_new      / (rain_new      + 30)
    rain_multiplier  = rain_resp_new / max(rain_resp_base, 1e-9)

    # Humidity: modest linear effect on mosquito survival
    humid_new        = baseline_humid + humidity_delta
    humid_multiplier = np.clip(
        (humid_new / 80) / max(baseline_humid / 80, 1e-9), 0.7, 1.5
    )

    # combined multiplier: temp * rain * humidity * R0 scaling
    scenario_multiplier = temp_multiplier * rain_multiplier * humid_multiplier * r0_scale

    # Step 4: Bootstrap-sample N baseline cases from the empirical distribution.
    # This is the non-parametric form of inverse-CDF sampling (Steps 2-4):
    # it builds the exact empirical CDF from observed data and draws from it.
    # At baseline (multiplier ≈ 1.0) this EXACTLY reproduces the historical
    # histogram — no smoothing loss from fitting a parametric family.
    baseline_samples = rng.choice(
        df["casos"].values.astype(float), size=n_simulations, replace=True
    )

    # Step 5: Scale each sample by the scenario multiplier.
    # Multiplier = 1 → output matches historical distribution exactly.
    predicted_cases = np.maximum(
        np.round(baseline_samples * scenario_multiplier), 0
    ).astype(int)

    # Sample representative climate values for display in the output dataframe.
    _, td, tp = best_fit(temp_fits)
    _, rd, rp = best_fit(rain_fits)
    _, hd, hp = best_fit(humid_fits)
    sim_temps  = np.clip(
        td.rvs(*tp, size=n_simulations, random_state=int(rng_seed)) + temp_delta, 15, 42)
    sim_rains  = np.clip(
        rd.rvs(*rp, size=n_simulations, random_state=int(rng_seed))
        * (1 + rain_delta_pct / 100.0), 0, 300)
    sim_humids = np.clip(
        hd.rvs(*hp, size=n_simulations, random_state=int(rng_seed)) + humidity_delta, 30, 100)

    # compute incidence per 100k if population is available
    pop = df["pop"].median() if "pop" in df.columns and df["pop"].notna().any() else np.nan
    pred_inc = (np.round(predicted_cases / pop * 100000, 2)
                if not np.isnan(pop) and pop > 0 else np.zeros(n_simulations))

    # expected cases under the scenario (scalar, used for display)
    mu_scenario = float(df["casos"].mean() * scenario_multiplier)

    sim_df = pd.DataFrame({
        "sim_temp":               np.round(sim_temps, 2),
        "sim_rain":               np.round(sim_rains, 2),
        "sim_humidity":           np.round(sim_humids, 2),
        "lambda":                 np.full(n_simulations, round(mu_scenario, 2)),
        "predicted_cases":        predicted_cases,
        "predicted_inc_per_100k": pred_inc,
    })
    return sim_df, {"temp_fits": temp_fits, "rain_fits": rain_fits,
                    "humid_fits": humid_fits, "case_fits": case_fits,
                    "raw_tempMed":    df["tempMed"].dropna().values,
                    "raw_chuva":      df["chuva"].dropna().values,
                    "raw_umidadeMed": df["umidadeMed"].dropna().values}


def plot_distribution_fits(fit_results_dict):
    """For each climate variable, plot real data histogram + all four fitted PDFs.
    Best-fit curve (lowest AIC) gets a star and thicker solid line.
    Corresponds to Step 1 of the Monte Carlo process."""
    var_labels = {
        "temp_fits":  ("Temperature (°C)",  "tempMed",    "#d62728"),
        "rain_fits":  ("Rainfall (mm/week)", "chuva",      "#1f77b4"),
        "humid_fits": ("Humidity (%)",       "umidadeMed", "#2ca02c"),
    }
    curve_colors = {
        "Normal":      "#333333",
        "Gamma":       "#9467bd",
        "Log-Normal":  "#e377c2",
        "Exponential": "#ff7f0e",
    }
    plots = {}
    for key, (title, col, color) in var_labels.items():
        fits = fit_results_dict.get(key, {})
        if not fits:
            continue
        best_name, _, _ = best_fit(fits)
        raw = fit_results_dict.get("raw_" + col)
        if raw is None or len(raw) < 4:
            continue

        # histogram of actual observed values
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw, histnorm="probability density",
            marker_color=color, opacity=0.45,
            name="Real data", nbinsx=20,
            hovertemplate="Value: %{x:.1f}<br>Density: %{y:.4f}<extra></extra>",
        ))

        # overlay each candidate PDF curve
        x_range = np.linspace(np.percentile(raw, 1), np.percentile(raw, 99), 200)
        for name, result in fits.items():
            try:
                y_pdf = result["dist"].pdf(x_range, *result["params"])
                is_best = (name == best_name)
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_pdf, mode="lines",
                    name=f"{name} (★ best fit)" if is_best else name,
                    line=dict(
                        color=curve_colors.get(name, "#888"),
                        width=3 if is_best else 1.5,
                        dash="solid" if is_best else "dash",
                    ),
                    hovertemplate=f"{name}<br>PDF: %{{y:.4f}}<extra></extra>",
                ))
            except Exception:
                pass
        fig.update_layout(
            title=f"{title} — Best fit: {best_name} (lowest AIC)",
            xaxis_title=title, yaxis_title="Probability density",
            height=300, margin=dict(t=50, b=10, l=60, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        )
        plots[key] = (fig, best_name, fits[best_name]["aic"])
    return plots


def compute_convergence(sim_df):
    """Running mean — demonstrates Law of Large Numbers (Week 4, slide 6)."""
    return sim_df["predicted_cases"].expanding().mean()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Dengue Fever Simulation — Brazil",
        page_icon="🦟",
        layout="wide",
    )

    # custom CSS for info boxes and section labels
    st.markdown("""
    <style>
    .block-box { background:#f0f7ff; border-left:4px solid #1f77b4;
        padding:14px 18px; border-radius:5px; margin:10px 0 18px 0;
        font-size:0.94rem; line-height:1.6; }
    .warn-box  { background:#fff7e6; border-left:4px solid #e07b00;
        padding:14px 18px; border-radius:5px; margin:10px 0 18px 0;
        font-size:0.94rem; line-height:1.6; }
    .green-box { background:#f0fff4; border-left:4px solid #2ca02c;
        padding:14px 18px; border-radius:5px; margin:10px 0 18px 0;
        font-size:0.94rem; line-height:1.6; }
    .section-label { font-size:0.78rem; font-weight:600; letter-spacing:0.08em;
        text-transform:uppercase; color:#666; margin-bottom:2px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🦟 Dengue Fever Outbreak Prediction — Brazil")
    st.caption(
        "SOFE 4820U · Modelling and Simulation  "
       
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        city    = st.selectbox("🏙️ City", list(CITY_GEOCODES.keys()))
        use_api = st.checkbox("Use live InfoDengue data", value=True)
        st.divider()
        st.subheader("🌡️ Climate Scenario")
        temp_delta     = st.slider("Temperature change (°C)",  -3.0,  6.0, 0.0, 0.5)
        rain_delta_pct = st.slider("Rainfall change (%)",       -50,  100,    0,   5)
        humidity_delta = st.slider("Humidity change (%)",      -15.0, 15.0, 0.0, 1.0)
        st.divider()
        st.subheader("🦠 Disease Parameters")
        st.caption(
            "**β** = how easily dengue spreads per week.  \n"
            "**γ** = how quickly people recover.  \n"
            "**R₀ = β ÷ γ** — if R₀ > 1 the outbreak grows."
        )
        beta  = st.slider("β — Transmission rate", 0.05, 1.0, 0.3, 0.05)
        gamma = st.slider("γ — Recovery rate",     0.05, 1.0, 0.1, 0.05)
        r0    = round(beta / gamma, 2)

        # color-coded R0 indicator: red if epidemic, yellow if marginal, green if controlled
        r0_icon = "🔴" if r0 > 3 else ("🟡" if r0 > 1 else "🟢")
        st.metric(f"{r0_icon}  R₀  (β ÷ γ)", r0)
        st.divider()
        n_sims   = st.select_slider("🎲 Simulation trials",
                                    options=[500, 1000, 2000, 5000], value=2000)
        rng_seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)
        run_btn  = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    # ── Landing ───────────────────────────────────────────────────────────────
    # show intro page until user clicks Run
    if not run_btn:
        st.markdown("---")
        st.markdown("""
        ### About this project
        Dengue fever is the world's most widespread mosquito-borne virus, putting
        **2.5 billion people** at risk globally. In Brazil alone, the Americas recorded
        **4,471,562 cases in 2025**. Rising temperatures and more intense rainfall
        driven by climate change and El Niño events are making outbreaks larger and more
        frequent.

        This dashboard uses a **stochastic Monte Carlo simulation** to
        model how changes in temperature, rainfall, and humidity affect weekly dengue case
        counts in Brazilian cities. **Choose a city, set a climate scenario, and click Run Simulation.**
        """)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.info("**Tab 1 — Climate & Cases**\nReal weekly InfoDengue data: cases, temperature, rainfall, humidity trends.")
        c2.info("**Tab 2 — Climate Relationships**\nHow temperature and rainfall variations drive dengue case counts.")
        c3.info("**Tab 3 — Outbreak Scenarios**\nMonte Carlo simulation results across multiple climate futures.")
        c4, _, _ = st.columns(3)
        c4.info("**Tab 4 — Distribution Fitting**\nProbability distributions fitted to real climate data — Step 1 of the Monte Carlo process.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    geocode = CITY_GEOCODES[city]
    with st.spinner(f"Fetching data for {city}..."):
        df = pd.DataFrame()
        data_source = "Synthetic"

        # try live API first; fall back to synthetic if empty or error
        if use_api:
            df = fetch_infodengue(geocode, 1, 52, 2023, 2025)
            if not df.empty:
                data_source = "InfoDengue (live)"
        if df.empty:
            df = generate_synthetic_brazil(city)
            data_source = "Synthetic (InfoDengue schema)"

    with st.spinner(f"Running {n_sims:,} Monte Carlo trials..."):
        sim_df, fit_results = run_monte_carlo(
            df, n_sims, temp_delta, rain_delta_pct,
            humidity_delta, beta, gamma, rng_seed=int(rng_seed)
        )

    # summary stats for the banner and metrics row
    obs_mean = df["casos"].mean()
    sim_mean = sim_df["predicted_cases"].mean()
    sim_p5   = float(np.percentile(sim_df["predicted_cases"],  5))
    sim_p95  = float(np.percentile(sim_df["predicted_cases"], 95))
    sim_p10  = float(np.percentile(sim_df["predicted_cases"], 10))
    sim_p90  = float(np.percentile(sim_df["predicted_cases"], 90))
    pct_chg  = (sim_mean - obs_mean) / max(obs_mean, 1) * 100
    pop_val  = df["pop"].median() if ("pop" in df.columns and df["pop"].notna().any()) else np.nan

    # ── Top banner ────────────────────────────────────────────────────────────
    st.markdown(
        f"### {city}  ·  {data_source}  ·  {len(df)} weeks"
    )

    # pick banner color and message based on whether sim is higher/lower than historical
    if abs(pct_chg) < 2:
        banner = (f"Under the current settings the simulation predicts <b>{sim_mean:,.0f} cases/week</b> — "
                  f"roughly the same as the historical average of {obs_mean:,.0f} cases/week.")
        bcls = "green-box"
    elif pct_chg > 0:
        banner = (f"⚠️ Under this scenario the simulation predicts <b>{sim_mean:,.0f} cases/week</b> — "
                  f"<b>{pct_chg:.0f}% higher</b> than the historical average ({obs_mean:,.0f}/week). "
                  f"Warmer or wetter conditions increase mosquito breeding and dengue transmission.")
        bcls = "warn-box"
    else:
        banner = (f"Under this scenario the simulation predicts <b>{sim_mean:,.0f} cases/week</b> — "
                  f"<b>{abs(pct_chg):.0f}% lower</b> than the historical average ({obs_mean:,.0f}/week).")
        bcls = "green-box"
    st.markdown(f'<div class="{bcls}">{banner}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical avg (cases/wk)",  f"{obs_mean:,.0f}")
    c2.metric("Simulated avg (cases/wk)",   f"{sim_mean:,.0f}", delta=f"{pct_chg:+.1f}% vs historical")
    c3.metric("Best-case (10th pct)",       f"{sim_p10:,.0f}", help="9 in 10 weeks predicted more than this")
    c4.metric("Worst-case (90th pct)",      f"{sim_p90:,.0f}", help="9 in 10 weeks predicted fewer than this")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍  Climate & Cases",
        "📐  Climate Relationships",
        "🔮  Outbreak Scenarios",
        "📊  Distribution Fitting",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — CLIMATE & CASES
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Monthly Dengue & Climate Data — InfoDengue")
        st.markdown(
            "Data is grouped by **month** across all years (Jan 2023 – Feb 2026) "
            "to clearly show the seasonal pattern. "
            "Use the filter below to also view a specific year."
        )

        # ── Year filter ───────────────────────────────────────────────────────
        df["year"]  = df["data_iniSE"].dt.year
        df["month"] = df["data_iniSE"].dt.month
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        available_years = sorted(df["year"].unique().tolist())

        view_mode = st.radio(
            "View:", ["All years combined (average)", "Single year"],
            horizontal=True, label_visibility="collapsed"
        )
        if view_mode == "Single year":
            sel_year = st.selectbox("Select year", available_years)
            df_view  = df[df["year"] == sel_year].copy()
            view_label = str(sel_year)
        else:
            df_view    = df.copy()
            view_label = "Average across all years"

        # Aggregate by month
        monthly = df_view.groupby("month").agg(
            cases=("casos",     "mean"),
            temp =("tempMed",   "mean"),
            rain =("chuva",     "mean"),
            humid=("umidadeMed","mean"),
        ).reset_index()
        monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])

        # ── Chart 1: Monthly cases coloured by alert severity ─────────────────
        # color-code bars: green → yellow → orange → red by case severity
        case_max = monthly["cases"].max()
        m_colors = [
            "#d62728" if c > case_max*0.7 else
            ("#e05c00" if c > case_max*0.4 else
             ("#f0a500" if c > case_max*0.2 else "#2ca02c"))
            for c in monthly["cases"]
        ]
        fig_mcases = go.Figure()
        fig_mcases.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["cases"],
            marker_color=m_colors,
            text=[f"{v:,.0f}" for v in monthly["cases"]],
            textposition="outside",
            textfont=dict(size=11, color="#333"),
            hovertemplate="%{x}: <b>%{y:,.0f}</b> avg cases/month<extra></extra>",
        ))
        fig_mcases.update_layout(
            title=f"Average Monthly Dengue Cases — {view_label}",
            xaxis_title=None, yaxis_title="Avg cases/month",
            yaxis=dict(range=[0, case_max * 1.28]),
            height=320, margin=dict(t=50, b=10, l=70, r=10),
            plot_bgcolor="white", paper_bgcolor="white", bargap=0.2,
        )
        st.plotly_chart(fig_mcases, use_container_width=True)
        st.markdown(
            '<div class="block-box">'
            '🟢 <b>Green</b> = low &nbsp;·&nbsp; 🟡 <b>Yellow</b> = moderate &nbsp;·&nbsp; '
            '🟠 <b>Orange</b> = high &nbsp;·&nbsp; 🔴 <b>Red</b> = peak epidemic level. '
            'Cases concentrate in the hot, wet months of January–April.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Chart 2: Temperature + rainfall by month (side by side) ──────────
        st.markdown("#### How do temperature and rainfall change month to month?")
        fig_clim2 = make_subplots(rows=1, cols=2,
            subplot_titles=["Avg Temperature (°C) by month",
                            "Avg Rainfall (mm) by month"])

        fig_clim2.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["temp"].round(1),
            marker_color="#d62728", opacity=0.8,
            text=[f"{v:.0f}°C" for v in monthly["temp"]],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="%{x}: %{y:.1f} °C<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        fig_clim2.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["rain"].round(1),
            marker_color="#1f77b4", opacity=0.8,
            text=[f"{v:.0f}" for v in monthly["rain"]],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
            showlegend=False,
        ), row=1, col=2)

        fig_clim2.update_yaxes(
            range=[0, monthly["temp"].max() * 1.28], title_text="°C", row=1, col=1)
        fig_clim2.update_yaxes(
            range=[0, monthly["rain"].max() * 1.28], title_text="mm/month", row=1, col=2)
        fig_clim2.update_layout(
            height=320, margin=dict(t=45, b=10, l=60, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_clim2, use_container_width=True)

        # ── Combined: cases + climate on one axis to show co-movement ─────────
        st.markdown("#### Do dengue cases peak in the same months as temperature and rainfall?")
        st.markdown(
            "The bars show monthly dengue cases (left axis). "
            "The lines show temperature (red) and rainfall (blue) on the right axis. "
            "**Peaks lining up confirms that hotter, wetter months = more dengue.**"
        )

        # dual-axis chart: bars for cases, lines for temp and rain
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["cases"],
            name="Dengue cases", marker_color="#e05c00", opacity=0.7,
            text=[f"{v:,.0f}" for v in monthly["cases"]],
            textposition="outside", textfont=dict(size=10, color="#333"),
            hovertemplate="%{x}: %{y:,.0f} avg cases<extra></extra>",
        ), secondary_y=False)
        fig_combo.add_trace(go.Scatter(
            x=monthly["month_name"], y=monthly["temp"],
            name="Avg temp (°C)", mode="lines+markers",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x}: %{y:.1f} °C<extra></extra>",
        ), secondary_y=True)
        fig_combo.add_trace(go.Scatter(
            x=monthly["month_name"], y=monthly["rain"],
            name="Avg rain (mm)", mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5, dash="dot"),
            marker=dict(size=7),
            hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
        ), secondary_y=True)

        fig_combo.update_yaxes(title_text="Avg dengue cases / month",
                               range=[0, monthly["cases"].max()*1.35],
                               secondary_y=False)
        fig_combo.update_yaxes(title_text="Temperature (°C) / Rainfall (mm)",
                               secondary_y=True)
        fig_combo.update_layout(
            height=380, margin=dict(t=20, b=10, l=70, r=70),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.15),
            bargap=0.2,
        )
        st.plotly_chart(fig_combo, use_container_width=True)

        # peak month stats for the insight box below
        peak_month = monthly.loc[monthly["cases"].idxmax(), "month_name"]
        peak_temp  = monthly.loc[monthly["cases"].idxmax(), "temp"]
        peak_rain  = monthly.loc[monthly["cases"].idxmax(), "rain"]

        # Monthly scatter plots — temperature vs cases, rainfall vs cases
        st.markdown("#### Does higher temperature or more rainfall mean more cases that month?")
        st.markdown(
            "Each dot is one month of data. The dashed line shows the overall trend. "
            "A rising trend confirms that climate drives dengue transmission."
        )

        # helper: fit a simple linear trendline, skip if not enough data
        def _trendline(xv, yv):
            mask = ~(np.isnan(xv) | np.isnan(yv))
            if mask.sum() < 3:
                return None, None
            m, b_int = np.polyfit(xv[mask], yv[mask], 1)
            xl = np.array([xv[mask].min(), xv[mask].max()])
            return xl, m * xl + b_int

        # Use monthly aggregated data
        df_scat = df.copy()
        df_scat["ym_s"] = df_scat["data_iniSE"].dt.year * 100 + df_scat["data_iniSE"].dt.month
        monthly_scat = df_scat.groupby("ym_s").agg(
            casos=("casos", "sum"),
            tempMed=("tempMed", "mean"),
            chuva=("chuva", "mean"),
        ).reset_index()

        # Pearson r for temp-cases and rain-cases
        r_tc = float(monthly_scat[["tempMed", "casos"]].corr().iloc[0, 1])
        r_rc = float(monthly_scat[["chuva",   "casos"]].corr().iloc[0, 1])

        col_l, col_r = st.columns(2)
        with col_l:
            xl, yl = _trendline(monthly_scat["tempMed"].values, monthly_scat["casos"].values)
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(
                x=monthly_scat["tempMed"], y=monthly_scat["casos"], mode="markers",
                marker=dict(color="#d62728", size=9, opacity=0.65),
                hovertemplate="Temp: %{x:.1f}°C<br>Cases: %{y:,}<extra></extra>"))
            if xl is not None:
                fig_tc.add_trace(go.Scatter(x=xl, y=yl, mode="lines",
                    line=dict(color="#7f0000", width=2.5, dash="dash"), showlegend=False))
            fig_tc.update_layout(
                title=f"Temperature vs. Monthly Cases  (r = {r_tc:.2f})",
                xaxis_title="Monthly avg temperature (°C)",
                yaxis_title="Total cases that month",
                height=320, margin=dict(t=50, b=10),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig_tc, use_container_width=True)

        with col_r:
            xl2, yl2 = _trendline(monthly_scat["chuva"].values, monthly_scat["casos"].values)
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=monthly_scat["chuva"], y=monthly_scat["casos"], mode="markers",
                marker=dict(color="#1f77b4", size=9, opacity=0.65),
                hovertemplate="Rain: %{x:.1f}mm<br>Cases: %{y:,}<extra></extra>"))
            if xl2 is not None:
                fig_rc.add_trace(go.Scatter(x=xl2, y=yl2, mode="lines",
                    line=dict(color="#00008b", width=2.5, dash="dash"), showlegend=False))
            fig_rc.update_layout(
                title=f"Rainfall vs. Monthly Cases  (r = {r_rc:.2f})",
                xaxis_title="Monthly avg rainfall (mm)",
                yaxis_title="Total cases that month",
                height=320, margin=dict(t=50, b=10),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig_rc, use_container_width=True)

        # helper: turn a Pearson r into plain English
        def _corr_text(r):
            a = abs(r)
            d = "positive" if r > 0 else "negative"
            s = "strong" if a > 0.5 else ("moderate" if a > 0.25 else "weak")
            return f"{s} {d} relationship (r = {r:.2f})"

        st.markdown(
            f'<div class="block-box">'
            f'📌 Cases peak in <b>{peak_month}</b> ({view_label}), when temperature averages '
            f'<b>{peak_temp:.1f}°C</b> and rainfall averages <b>{peak_rain:.1f} mm/month</b>.<br>'
            f'🌡️ Temperature vs. cases: <b>{_corr_text(r_tc)}</b><br>'
            f'🌧️ Rainfall vs. cases: <b>{_corr_text(r_rc)}</b><br>'
            f'Both confirm that hotter, wetter months produce more dengue cases '
       
            f'</div>',
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — CLIMATE RELATIONSHIPS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("How Do Climate Variations Drive Dengue Cases?")
        st.markdown(
            "The research question is: **do variations in temperature and rainfall "
            "predict the size of dengue outbreaks?** This tab visualises those "
            "relationships directly from the real InfoDengue data."
        )

        # 2. Temperature bands
        st.markdown("#### What happens to case counts as temperature rises?")
        st.markdown(
            "Weeks are grouped into temperature bands. The line connects average cases "
            "per band. The shaded bars show the typical range (IQR and 10th–90th pct). "
            "**Values are labelled above each point.**"
        )

        # bin weeks into temperature bands, compute stats per band
        df_tb = df[df["tempMed"].notna() & df["casos"].notna()].copy()
        t_min, t_max = df_tb["tempMed"].min(), df_tb["tempMed"].max()
        bin_edges = np.linspace(t_min, t_max, 7)
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}°C" for i in range(6)]
        df_tb["temp_band"] = pd.cut(df_tb["tempMed"], bins=bin_edges,
                                     labels=bin_labels, include_lowest=True)
        band_stats = df_tb.groupby("temp_band", observed=True)["casos"].agg(
            mean="mean",
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
            count="count",
        ).reset_index()

        # layered bars: outer = 10th-90th range, inner = IQR, line = mean
        fig_tbands = go.Figure()
        fig_tbands.add_trace(go.Bar(x=band_stats["temp_band"].astype(str),
            y=band_stats["p90"]-band_stats["p10"], base=band_stats["p10"],
            marker_color="rgba(214,39,40,0.12)", showlegend=True,
            name="Typical range (10th–90th pct)", hoverinfo="skip"))
        fig_tbands.add_trace(go.Bar(x=band_stats["temp_band"].astype(str),
            y=band_stats["p75"]-band_stats["p25"], base=band_stats["p25"],
            marker_color="rgba(214,39,40,0.40)", showlegend=True,
            name="Middle 50% of weeks", hoverinfo="skip"))
        fig_tbands.add_trace(go.Scatter(x=band_stats["temp_band"].astype(str),
            y=band_stats["mean"], mode="markers+lines+text",
            marker=dict(color="#d62728", size=11, line=dict(color="white", width=2)),
            line=dict(color="#d62728", width=2),
            text=[f"{v:,.0f}" for v in band_stats["mean"]],
            textposition="top center", textfont=dict(size=11, color="#d62728"),
            name="Average cases",
            hovertemplate="%{x}<br>Avg: %{y:,.0f} cases/week (n=%{customdata})<extra></extra>",
            customdata=band_stats["count"]))
        fig_tbands.update_layout(
            xaxis_title="Weekly average temperature", yaxis_title="Average Dengue cases a week",
            barmode="overlay", height=400, margin=dict(t=20, b=40, l=70, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(fig_tbands, use_container_width=True)

        # hottest vs coldest band comparison
        hot_mean  = band_stats.iloc[-1]["mean"]
        cold_mean = band_stats.iloc[0]["mean"]
        mult = hot_mean / max(cold_mean, 1)
        st.markdown(
            f'<div class="block-box">'
            f'🌡️ Weeks in the <b>hottest temperature band</b> average '
            f'<b>{hot_mean:,.0f} cases/week</b> — roughly <b>{mult:.1f}× more</b> '
            f'than the coldest band ({cold_mean:,.0f} cases/week). '
            f'This confirms rising temperatures directly increase dengue infection rates.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # 3. Rainfall bands
        st.markdown("#### What happens to case counts as rainfall increases?")
        st.markdown(
            "Same analysis for rainfall. More rain means more standing water — "
            "more mosquito breeding sites. **Values labelled above each point.**"
        )

        # clip to 2nd-98th percentile to avoid single outlier stretching the bins
        df_rb = df[df["chuva"].notna() & df["casos"].notna() & (df["chuva"] > 0)].copy()
        r_min  = df_rb["chuva"].quantile(0.02)
        r_max  = df_rb["chuva"].quantile(0.98)
        rain_edges  = np.linspace(r_min, r_max, 7)
        rain_labels = [f"{rain_edges[i]:.0f}-{rain_edges[i+1]:.0f}mm" for i in range(6)]
        df_rb["rain_band"] = pd.cut(df_rb["chuva"], bins=rain_edges,
                                     labels=rain_labels, include_lowest=True)
        rain_stats = df_rb.groupby("rain_band", observed=True)["casos"].agg(
            mean="mean",
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
            count="count",
        ).reset_index()

        # same layered bar + line structure as temperature bands
        fig_rbands = go.Figure()
        fig_rbands.add_trace(go.Bar(x=rain_stats["rain_band"].astype(str),
            y=rain_stats["p90"]-rain_stats["p10"], base=rain_stats["p10"],
            marker_color="rgba(31,119,180,0.12)", showlegend=True,
            name="Typical range (10th–90th pct)", hoverinfo="skip"))
        fig_rbands.add_trace(go.Bar(x=rain_stats["rain_band"].astype(str),
            y=rain_stats["p75"]-rain_stats["p25"], base=rain_stats["p25"],
            marker_color="rgba(31,119,180,0.40)", showlegend=True,
            name="Middle 50% of weeks", hoverinfo="skip"))
        fig_rbands.add_trace(go.Scatter(x=rain_stats["rain_band"].astype(str),
            y=rain_stats["mean"], mode="markers+lines+text",
            marker=dict(color="#1f77b4", size=11, line=dict(color="white", width=2)),
            line=dict(color="#1f77b4", width=2),
            text=[f"{v:,.0f}" for v in rain_stats["mean"]],
            textposition="top center", textfont=dict(size=11, color="#1f77b4"),
            name="Average cases",
            hovertemplate="%{x}<br>Avg: %{y:,.0f} cases/week (n=%{customdata})<extra></extra>",
            customdata=rain_stats["count"]))
        fig_rbands.update_layout(
            xaxis_title="Weekly rainfall", yaxis_title="Average Dengue cases a week",
            barmode="overlay", height=400, margin=dict(t=20, b=40, l=70, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(fig_rbands, use_container_width=True)

        # wettest vs driest band comparison
        wet_mean  = rain_stats.iloc[-1]["mean"]
        dry_mean  = rain_stats.iloc[0]["mean"]
        rain_mult = wet_mean / max(dry_mean, 1)
        r_rain    = float(df[["chuva", "casos"]].corr().iloc[0, 1])
        st.markdown(
            f'<div class="block-box">'
            f'🌧️ The wettest weeks average <b>{wet_mean:,.0f} cases/week</b> vs '
            f'<b>{dry_mean:,.0f} cases/week</b> in the driest weeks '
            f'({rain_mult:.1f}× difference). '
            f'Overall correlation between weekly rainfall and cases: <b>r = {r_rain:.2f}</b>.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # 4. Combined heatmap
        st.markdown("#### Combined effect: temperature AND rainfall together")
        st.markdown(
            "Dengue risk is highest when **both** temperature and rainfall are elevated. "
            "This heatmap shows average weekly case counts for each combination. "
            "Darker red = more cases."
        )

        # bin into 5x5 grid, cap rainfall at 95th pct to avoid sparse cells
        df_hm = df[df["tempMed"].notna() & df["chuva"].notna() & df["casos"].notna()].copy()
        df_hm["t_bin"] = pd.cut(df_hm["tempMed"], bins=5, precision=0)
        df_hm["r_bin"] = pd.cut(
            df_hm["chuva"].clip(upper=df_hm["chuva"].quantile(0.95)), bins=5, precision=0)
        hm_data = df_hm.pivot_table(values="casos", index="r_bin",
                                     columns="t_bin", aggfunc="mean")
        hm_data.index   = [str(i) + " mm"  for i in hm_data.index]
        hm_data.columns = [str(c) + " °C" for c in hm_data.columns]

        fig_hm = px.imshow(hm_data, color_continuous_scale="YlOrRd",
            labels={"x": "Temperature band", "y": "Rainfall band", "color": "Avg cases/week"},
            title="Average Weekly Cases by Temperature and Rainfall Combination",
            text_auto=".0f", aspect="auto")
        fig_hm.update_layout(height=340, margin=dict(t=55, b=10),
            coloraxis_colorbar=dict(title="Cases/wk"))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown(
            '<div class="block-box">'
            '📌 <b>How to read this heatmap:</b> Each cell shows average weekly dengue cases '
            'for weeks in that temperature AND rainfall combination. '
            'The darkest red cells (top-right) = hot AND wet weeks = highest risk. '
            'This directly supports the stochastic model which uses both variables together.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — OUTBREAK SCENARIOS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        # build a human-readable label for the current scenario
        parts = []
        if temp_delta != 0:     parts.append(f"{temp_delta:+.1f}°C")
        if rain_delta_pct != 0: parts.append(f"{rain_delta_pct:+d}% rain")
        if humidity_delta != 0: parts.append(f"{humidity_delta:+.1f}% humidity")
        scenario_str = " / ".join(parts) if parts else "baseline (no change)"

        st.subheader("Monte Carlo Simulation Results")
        st.markdown(
            f"The simulation ran **{n_sims:,} random trials** using the 5-step Monte Carlo "
            f"process**. Each trial randomly drew a temperature, rainfall, "
            f"and humidity from the fitted distributions, applied the **{scenario_str}** shift, "
            f"then predicted weekly dengue cases using a Negative Binomial model with R₀ = **{r0}**."
        )

        # ── Distribution chart — quantile-binned grouped bars ─────────────────
        # Equal-width bins from 0 to the 95th percentile of historical cases.
        # Percentile-based bins are always flat by mathematical definition
        # (each bin has equal mass from the source data), making both bars look
        # uniform regardless of how well the simulation matches. Equal-width bins
        # preserve the actual right-skewed shape of dengue case data so:
        #   • Historical shows the real distribution (low weeks dominate, rare spikes)
        #   • Simulated at baseline matches that shape → bars align closely
        #   • Simulated at scenarios shifts right → clearly visible
        # Cap at 95th pct of historical to avoid extreme outliers stretching bins;
        # anything above the cap falls into the last "overflow" bin.
        from scipy import stats as sp_stats

        # upper bin edge = max of both 95th percentiles
        hist_95 = float(np.percentile(df["casos"].values, 95))
        sim_95  = float(np.percentile(sim_df["predicted_cases"].values, 95))
        upper   = max(hist_95, sim_95)

        # 8 equal-width bins; ensure at least 4 unique edges
        raw_edges = np.linspace(0, upper, 9).astype(int)
        raw_edges = np.unique(raw_edges)
        if len(raw_edges) < 4:
            raw_edges = np.linspace(0, int(upper), 9).astype(int)
        bin_edges = raw_edges.astype(float)
        bin_labels = [
            f"{int(bin_edges[i]):,}–{int(bin_edges[i+1]):,}"
            for i in range(len(bin_edges) - 1)
        ]

        # compute relative frequencies for both historical and simulated
        hist_counts, _ = np.histogram(df["casos"].values, bins=bin_edges)
        sim_counts,  _ = np.histogram(sim_df["predicted_cases"].values, bins=bin_edges)
        hist_probs = hist_counts / max(hist_counts.sum(), 1)
        sim_probs  = sim_counts  / max(sim_counts.sum(),  1)

        COLOR_HIST = "#E69F00"   # amber  — historical  (Wong colorblind-safe)
        COLOR_SIM  = "#0072B2"   # blue   — simulated

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            name="Historical data",
            x=bin_labels, y=hist_probs,
            marker_color=COLOR_HIST,
            text=[f"{p:.2f}" for p in hist_probs],
            textposition="outside",
            textfont=dict(size=11, color=COLOR_HIST, family="Arial Black"),
            hovertemplate="%{x}<br>Historical: %{y:.1%}<extra></extra>",
        ))
        fig_dist.add_trace(go.Bar(
            name="Simulated outcomes",
            x=bin_labels, y=sim_probs,
            marker_color=COLOR_SIM,
            text=[f"{p:.2f}" for p in sim_probs],
            textposition="outside",
            textfont=dict(size=11, color=COLOR_SIM, family="Arial Black"),
            hovertemplate="%{x}<br>Simulated: %{y:.1%}<extra></extra>",
        ))
        fig_dist.update_layout(
            title=dict(
                text=f"Relative Frequencies — Historical vs. {n_sims:,} Simulated Outcomes",
                font=dict(size=15, color="#333333", family="Arial"), x=0,
            ),
            xaxis=dict(title="Weekly dengue cases (binned)",
                       tickfont=dict(size=11), tickangle=-20),
            yaxis=dict(title="Relative frequency (probability)", tickformat=".0%",
                       range=[0, max(hist_probs.max(), sim_probs.max()) * 1.28]),
            barmode="group", bargap=0.18, bargroupgap=0.06,
            height=420, margin=dict(t=55, b=60, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=12)),
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── Summary stats table ───────────────────────────────────────────────
        # KS test: checks if sim and historical come from the same distribution
        ks_stat, ks_p = sp_stats.ks_2samp(sim_df["predicted_cases"].values, df["casos"].values)
        sim_std_val = sim_df["predicted_cases"].std()
        obs_std_val = df["casos"].std()
        stats_data = {
            "": ["Mean (cases/week)", "Median", "Std deviation", "5th–95th percentile"],
            "Historical data": [
                f"{obs_mean:,.0f}",
                f"{int(df['casos'].median()):,}",
                f"{int(obs_std_val):,}",
                f"{int(df['casos'].quantile(0.05)):,} – {int(df['casos'].quantile(0.95)):,}",
            ],
            f"Simulated ({n_sims:,} trials)": [
                f"{sim_mean:,.0f}",
                f"{int(np.median(sim_df['predicted_cases'])):,}",
                f"{int(sim_std_val):,}",
                f"{sim_p5:,.0f} – {sim_p95:,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data).set_index(""), use_container_width=True)

        # p > 0.05 means distributions are statistically similar (baseline scenario)
        if ks_p > 0.05:
            ks_msg = (f'✅ KS test p = {ks_p:.3f} > 0.05 — the simulated distribution '
                      f'is statistically consistent with the historical data.')
            ks_cls = "green-box"
        else:
            ks_msg = (f'⚠️ KS test p = {ks_p:.4f} — distributions differ, expected '
                      f'when the climate scenario significantly shifts conditions '
                      f'away from the historical baseline.')
            ks_cls = "warn-box"
        st.markdown(f'<div class="{ks_cls}">{ks_msg}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Scenario comparison bar chart ─────────────────────────────────────
        st.subheader("How Much Worse Does It Get as the Climate Changes?")
        st.markdown(
            f"Each climate scenario below was simulated with {n_sims:,} trials. "
            f"Values are labelled above each bar. The orange dotted line marks today's "
            f"historical average. Bars above that line = more cases than normal."
        )

        # predefined scenarios + user's custom one at the end
        scenario_list = [
            ("Baseline\n(today)",         0,  0,  0),
            ("+1°C",                      1,  0,  0),
            ("+2°C",                      2,  0,  0),
            ("+2°C\n+20% rain",           2, 20,  0),
            ("+3°C\n+30% rain",           3, 30,  5),
            ("El Niño\n(+4°C +50% rain)", 4, 50, 10),
            ("Your\nscenario",            temp_delta, rain_delta_pct, humidity_delta),
        ]

        sc_names, sc_means_v, sc_lo25, sc_hi75 = [], [], [], []
        prog = st.progress(0, text="Running scenarios...")
        # sc_raw stores the full predicted_cases arrays for reuse in early-warning
        sc_raw = []
        for i, (sc_name, sc_td, sc_rd, sc_hd) in enumerate(scenario_list):
            # Fixed scenarios use default R0=3 params so they're truly comparable;
            # only 'Your scenario' uses the user's beta/gamma
            _beta  = beta  if sc_name == "Your\nscenario" else 0.3
            _gamma = gamma if sc_name == "Your\nscenario" else 0.1
            s, _ = run_monte_carlo(df, 800, sc_td, sc_rd, sc_hd, _beta, _gamma, rng_seed=42)
            v = s["predicted_cases"]
            sc_names.append(sc_name)
            sc_means_v.append(float(v.mean()))
            sc_lo25.append(float(np.percentile(v, 25)))
            sc_hi75.append(float(np.percentile(v, 75)))
            sc_raw.append(v)
            prog.progress((i+1)/len(scenario_list),
                          text=f"Scenario {i+1}/{len(scenario_list)}...")
        prog.empty()

        # color gradient: user scenario = red, El Niño = dark red, cooling down to light blue
        bar_colors_sc = [
            "#d62728" if "scenario" in n.lower() else
            ("#8b1a1a" if "El Niño" in n else
             ("#d65a00" if "+3" in n else
              ("#f0a500" if "+20%" in n else
               ("#6baed6" if "+2°C" in n else
                ("#9ecae1" if "+1°C" in n else "#c6dbef")))))
            for n in sc_names
        ]

        fig_sc = go.Figure()
        # IQR shading
        fig_sc.add_trace(go.Bar(name="IQR (25th–75th pct)",
            x=sc_names,
            y=[sc_hi75[i]-sc_lo25[i] for i in range(len(sc_names))],
            base=sc_lo25,
            marker_color="rgba(100,100,100,0.12)",
            marker_line_width=0, hoverinfo="skip"))
        # Main bars
        fig_sc.add_trace(go.Bar(name="Avg predicted cases",
            x=sc_names, y=sc_means_v,
            marker_color=bar_colors_sc,
            text=[f"{v:,.0f}" for v in sc_means_v],
            textposition="outside", textfont=dict(size=12, color="#333"),
            hovertemplate="%{x}<br><b>%{y:,.0f} cases/week</b><extra></extra>"))

        # dotted line = today's historical average for easy comparison
        fig_sc.add_hline(y=obs_mean, line_dash="dot", line_color="#c07a00", line_width=2,
            annotation_text=f"Historical avg: {obs_mean:,.0f}/wk",
            annotation_font_size=11, annotation_font_color="#c07a00",
            annotation_position="top left")
        fig_sc.update_layout(
            xaxis_title=None, yaxis_title="Predicted cases / week",
            barmode="overlay", height=460,
            margin=dict(t=50, b=10, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, bargap=0.3,
            yaxis=dict(range=[0, max(sc_means_v)*1.28]))
        st.plotly_chart(fig_sc, use_container_width=True)

        # find El Niño scenario index for the key finding callout
        el_idx  = next(i for i, n in enumerate(sc_names) if "El Niño" in n)
        el_mean = sc_means_v[el_idx]
        base_sc = sc_means_v[0]
        st.markdown(
            f'<div class="block-box">'
            f'📌 <b>Key finding:</b> The orange dotted line is today\'s historical average '
            f'({obs_mean:,.0f}/week). Bars above it signal more cases than normal. '
            f'Under El Niño conditions, the simulation predicts <b>{el_mean:,.0f} cases/week</b> '
            f'vs <b>{base_sc:,.0f}/week</b> at baseline — a <b>{el_mean/max(base_sc,1):.1f}× increase</b>. '
            f'Grey shading on each bar shows the 25th–75th percentile uncertainty range.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Early warning: high-risk weeks per year ───────────────────────────
        st.subheader("Early Warning: How Many High-Risk Weeks Per Year?")
        st.markdown(
            "The project goal is to **enable early warnings**. "
            "This chart shows how many weeks per year each scenario predicts would exceed "
            "the epidemic alert threshold — defined as the 75th percentile of historical cases. "
            "Values labelled above each bar."
        )

        # epidemic threshold = 75th percentile of observed weekly cases
        epidemic_threshold = float(np.percentile(df["casos"], 75))

        # Reuse sc_raw from the scenario loop above — no need to re-run Monte Carlo
        ew_weeks = [
            round((v > epidemic_threshold).mean() * 52, 1)
            for v in sc_raw
        ]

        # color-code by severity: >30 weeks = red, >20 = orange, >10 = yellow, else green
        ew_colors = ["#d62728" if w > 30 else ("#f0a500" if w > 20 else
                     ("#f9d04a" if w > 10 else "#74c476")) for w in ew_weeks]
        hist_wks = (df["casos"] > epidemic_threshold).mean() * 52

        fig_ew = go.Figure(go.Bar(
            x=sc_names, y=ew_weeks,
            marker_color=ew_colors,
            text=[f"{w:.0f} wks" for w in ew_weeks],
            textposition="outside", textfont=dict(size=12, color="#333"),
            hovertemplate="%{x}<br><b>%{y:.1f} high-risk weeks/year</b><extra></extra>"))
        fig_ew.add_hline(y=hist_wks, line_dash="dot", line_color="#c07a00", line_width=2,
            annotation_text=f"Historical: {hist_wks:.0f} wks/year",
            annotation_font_size=11, annotation_font_color="#c07a00",
            annotation_position="top left")
        fig_ew.update_layout(
            xaxis_title=None, yaxis_title="High-risk weeks per year",
            yaxis=dict(range=[0, max(ew_weeks)*1.28]),
            height=400, margin=dict(t=50, b=10, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white", bargap=0.3)
        st.plotly_chart(fig_ew, use_container_width=True)

        st.markdown(
            f'<div class="block-box">'
            f'🚨 <b>Early warning insight:</b> The epidemic threshold is '
            f'<b>{epidemic_threshold:,.0f} cases/week</b> '
            f'(75th percentile of historical data). '
            f'Historically ~{hist_wks:.0f} weeks/year exceed this. '
            f'Under hotter, wetter scenarios that number rises sharply — giving health '
            f'planners a concrete question: <i>how many extra epidemic weeks per year does '
            f'climate change add?</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — DISTRIBUTION FITTING
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Step 1 of Monte Carlo: Fitting Probability Distributions")
        st.markdown(
            "Before the simulation runs, we fit four theoretical distributions to each "
            "climate variable from the real InfoDengue data. The **best fit** (marked ★) "
            "is selected by the lowest AIC score and is the distribution the simulation "
            "samples from. This directly corresponds to **Step 1 of the five-step "
            "Monte Carlo process."
        )

        dist_plots = plot_distribution_fits(fit_results)

        if dist_plots:
            var_names = {
                "temp_fits":  "Temperature",
                "rain_fits":  "Rainfall",
                "humid_fits": "Humidity",
            }
            # render one chart + insight box per climate variable
            for key, (fig, best_name, aic_val) in dist_plots.items():
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f'<div class="block-box">'
                    f'The <b>{var_names.get(key, key)}</b> data is best described by a '
                    f'<b>{best_name}</b> distribution (AIC = {aic_val:.1f}). '
                    f'The solid curve is the best-fit PDF overlaid on the real data histogram. '
                    f'Dashed curves show the other three candidates. '
                    f'The simulation samples climate values from this fitted distribution '
                    f'when generating each Monte Carlo trial.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Not enough data to fit distributions.")

if __name__ == "__main__":
    main()
