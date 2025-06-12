"""
Streamlit UI for bearing-only resection (auto‑zoom)
==================================================
*Update — the Leaflet map now *centres and zooms* automatically on the
estimated location once the solver runs.*

Run with:
```bash
streamlit run resection_app.py
```
"""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import LayerControl, Marker, Polygon, TileLayer
import numpy as np
from numpy.linalg import eigh
from scipy.stats import chi2
from pyproj import Transformer

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# try import backend
try:
    from src import solve_resection_odr
except ImportError:
    st.error("Could not import solve_resection_odr. Ensure the module is in PYTHONPATH.")
    st.stop()

# -------------------------------------------------- helpers

def utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    hemi = "326" if lat >= 0 else "327"
    return f"EPSG:{hemi}{zone:02d}"

def to_utm(lon, lat):
    crs = utm_crs(float(lon), float(lat))
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return tr.transform(lon, lat), tr


def to_latlon(x, y, tr):
    return tr.transform(x, y, direction="INVERSE")


def ellipse_coords(cov, centre_xy, tr, alpha=0.95, n=72):
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k2 = chi2.ppf(alpha, df=2)
    a, b = np.sqrt(k2 * vals)
    phi = np.arctan2(vecs[1, 0], vecs[0, 0])
    t = np.linspace(0, 2 * np.pi, n)
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    pts = R @ np.vstack((xs, ys)) + centre_xy[:, None]
    latlon = [to_latlon(pts[0, i], pts[1, i], tr)[::-1] for i in range(n)]  # folium wants [lat, lon]
    return latlon

# -------------------------------------------------- Streamlit

st.set_page_config(page_title="Bearing‑only Resection", layout="wide")
st.title("Bearing‑only Resection demo (auto‑zoom)")

# — sidebar inputs
if "rows" not in st.session_state:
    st.session_state.rows = 3

rows = st.session_state.rows
lats, lons, brgs = [], [], []
st.sidebar.markdown("### Anchor observations")

for i in range(rows):
    c1, c2, c3 = st.sidebar.columns(3)
    lat = c1.number_input("lat", key=f"lat{i}", value=0.0, format="%.6f")
    lon = c2.number_input("lon", key=f"lon{i}", value=0.0, format="%.6f")
    brg = c3.number_input("bearing°", key=f"brg{i}", value=0.0, format="%.2f")
    lats.append(lat); lons.append(lon); brgs.append(brg)

add, rem = st.sidebar.columns(2)
if add.button("Add row"):
    st.session_state.rows += 1
    st.experimental_rerun()
if rem.button("Remove", disabled=rows<=2):
    st.session_state.rows -= 1
    st.experimental_rerun()

st.sidebar.markdown("### Noise")
σθ = st.sidebar.number_input("σθ (° rms)", value=0.5, min_value=0.0, step=0.1)
σA = st.sidebar.number_input("σ_anchor (m rms)", value=0.02, min_value=0.0, step=0.01)
robust = st.sidebar.checkbox("Huber robust", value=True)
run = st.sidebar.button("Solve position")

# — build folium map, default world view
m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
TileLayer("OpenStreetMap", name="OSM").add_to(m)
TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="&copy; Esri",
    name="Satellite",
).add_to(m)
LayerControl().add_to(m)

if run:
    lats_arr = np.array(lats)
    lons_arr = np.array(lons)
    # project
    (Ex, Ny), tr = to_utm(lons_arr, lats_arr)
    anchors = np.column_stack((Ex, Ny))
    sigma_theta = np.deg2rad(σθ)
    Sigma = np.repeat((σA**2 * np.eye(2))[None, :, :], anchors.shape[0], axis=0)

    res = solve_resection_odr(
        theta=np.deg2rad(brgs),
        anchors=anchors,
        sigma_theta=sigma_theta,
        Sigma=Sigma,
        robust=robust,
    )

    # convert back to lat/lon centre
    lat_c, lon_c = to_latlon(res["position"][0], res["position"][1], tr)
    m.location = [lat_c, lon_c]
    m.zoom_start = 17

    # anchors markers
    for lat, lon, b in zip(lats_arr, lons_arr, brgs):
        Marker([lat, lon], popup=f"Anchor<br>{lat:.5f},{lon:.5f}<br>{b:.1f}°").add_to(m)

    # solution marker
    Marker([lat_c, lon_c], icon=folium.Icon(color="red"), popup="Estimated position").add_to(m)

    # ellipse
    ell = ellipse_coords(res["cov"], res["position"], tr)
    Polygon(ell, color="red", weight=2, fill=True, fill_opacity=0.15).add_to(m)

    semi_axes = np.sqrt(chi2.ppf(0.95, 2) * eigh(res["cov"])[0][::-1])
    st.success(f"Solution: {lat_c:.6f}°, {lon_c:.6f}° – 95 % semi‑axes {semi_axes[0]:.2f} m × {semi_axes[1]:.2f} m")

st_folium(m, height=600, width=1200)
