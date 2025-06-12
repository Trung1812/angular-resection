"""Streamlit UI for bearing-only resection using Mapbox and Matplotlib."""
from __future__ import annotations

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from pyproj import Transformer

from src import solve_resection_odr, confidence_ellipse

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_dms_pattern = re.compile(
    r"(?P<deg>[-+]?\d+)(?:[°\s]+(?P<min>\d+))?(?:['\s]+(?P<sec>\d+(?:\.\d+)?))?"  # deg min sec
)


def parse_angle(text: str) -> float:
    """Parse a latitude or longitude in decimal or DMS format to decimal degrees."""
    text = text.strip()
    m = _dms_pattern.match(text)
    if m and (m.group("min") or m.group("sec")):
        deg = float(m.group("deg"))
        minutes = float(m.group("min") or 0)
        seconds = float(m.group("sec") or 0)
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)
    return float(text)


def utm_transformer(lon: float, lat: float) -> Transformer:
    """Return transformer for local UTM zone covering given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    hemi = "326" if lat >= 0 else "327"
    return Transformer.from_crs("EPSG:4326", f"EPSG:{hemi}{zone:02d}", always_xy=True)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Angular Resection", layout="wide")
st.title("Angular Resection demo")

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    st.warning("Set MAPBOX_TOKEN environment variable for tiles")

if "rows" not in st.session_state:
    st.session_state.rows = 2

rows = st.session_state.rows

st.sidebar.markdown("### Anchor observations")
lat_inputs = []
lon_inputs = []
bearing_inputs = []

for i in range(rows):
    c1, c2, c3 = st.sidebar.columns(3)
    lat_inputs.append(c1.text_input(f"lat {i+1}", value="0.0"))
    lon_inputs.append(c2.text_input(f"lon {i+1}", value="0.0"))
    bearing_inputs.append(c3.number_input(f"bearing° {i+1}", value=0.0))

c_add, c_rem = st.sidebar.columns(2)
if c_add.button("Add more point"):
    st.session_state.rows += 1
if c_rem.button("Remove last input", disabled=rows <= 2):
    st.session_state.rows -= 1

st.sidebar.markdown("### Noise")
sigma_theta_deg = st.sidebar.number_input("bearing noise σθ [deg]", value=0.5)
sigma_gnss = st.sidebar.number_input("GNSS noise σ [m]", value=0.02)
robust = st.sidebar.checkbox("Huber robust", value=True)

run = st.sidebar.button("Solve", type="primary")

# Placeholder for results
map_placeholder = st.empty()
plt_placeholder = st.empty()

if run:
    try:
        lats = np.array([parse_angle(v) for v in lat_inputs], dtype=float)
        lons = np.array([parse_angle(v) for v in lon_inputs], dtype=float)
    except ValueError as exc:
        st.error(f"Invalid latitude/longitude: {exc}")
        st.stop()

    bearings = np.deg2rad(np.array(bearing_inputs, dtype=float))

    transformer = utm_transformer(float(np.mean(lons)), float(np.mean(lats)))
    East, North = transformer.transform(lons, lats)
    anchors = np.column_stack((East, North))
    Sigma = np.repeat((sigma_gnss ** 2 * np.eye(2))[None, :, :], anchors.shape[0], axis=0)

    res = solve_resection_odr(
        theta=bearings,
        anchors=anchors,
        sigma_theta=np.deg2rad(sigma_theta_deg),
        Sigma=Sigma,
        robust=robust,
    )

    centre_lon, centre_lat = transformer.transform(res["position"][0], res["position"][1], direction="INVERSE")
    a, b, phi = confidence_ellipse(res["cov"])
    t = np.linspace(0, 2 * np.pi, 100)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    ellipse_xy = R @ np.vstack((a * np.cos(t), b * np.sin(t))) + res["position"][:, None]
    ell_lon, ell_lat = transformer.transform(ellipse_xy[0], ellipse_xy[1], direction="INVERSE")

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode="markers", marker=dict(size=8), name="Anchors"))
    fig.add_trace(go.Scattermapbox(lat=[centre_lat], lon=[centre_lon], mode="markers", marker=dict(size=10, color="red"), name="Solution"))
    fig.add_trace(go.Scattermapbox(lat=ell_lat, lon=ell_lon, mode="lines", fill="toself", line=dict(color="red"), name="95% ellipse"))
    fig.update_layout(
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style="streets",
            zoom=17,
            center=dict(lat=centre_lat, lon=centre_lon),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )

    map_placeholder.plotly_chart(fig, use_container_width=True)

    # Matplotlib figure in projected coordinates
    fig2, ax = plt.subplots()
    ax.scatter(anchors[:, 0], anchors[:, 1], label="Anchors")
    ax.scatter(res["position"][0], res["position"][1], c="r", label="Solution")
    ax.plot(ellipse_xy[0], ellipse_xy[1], "r-", label="95% ellipse")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.legend()
    plt_placeholder.pyplot(fig2)
