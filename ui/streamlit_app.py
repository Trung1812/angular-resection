"""
Interactive web UI (Streamlit) for 2D bearing-only resection
===========================================================
Uses the **solve_resection_odr** backend we built earlier and visualises the
solution and its 95% confidence ellipse on a Leaflet map (via **streamlit-folium**).

Requirements
------------
    pip install streamlit streamlit-folium folium numpy scipy pyproj

Run
----
    streamlit run resection_app.py

UI workflow
-----------
1. Enter anchors as latitude, longitude (decimal degrees) and the observed
   bearing **from the unknown position X toward the anchor** (degrees clockwise
   from true north).
2. Press *Solve position*.
3. The map updates showing:
    • anchor markers,
    • estimated position (red marker),
    • red 95 % confidence ellipse.

All computations are done in an appropriate UTM projection derived from the
anchor centroid, so metric accuracy is preserved.
"""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import TileLayer, LayerControl, Popup, Marker, PolyLine, Polygon
import numpy as np
from scipy.stats import chi2
from numpy.linalg import eigh
from pyproj import Transformer
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------------------------
# Import the solver from our earlier module (assumed in PYTHONPATH)
# -----------------------------------------------------------------------------
try:
    from src import solve_resection_odr
except ImportError as e:
    print(e.msg)
    st.error("Cannot import solve_resection_odr - make sure odr_resection_solver.py is on PYTHONPATH")
    st.stop()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def utm_crs_from_lon(lon: float, lat: float) -> str:
    """Return EPSG code for suitable UTM zone covering (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "326" if lat >= 0 else "327"  # 326 = WGS84 / UTM north, 327 south
    return f"EPSG:{hemisphere}{zone:02d}"


def to_utm(lons, lats):
    """Convert arrays of lon/lat to UTM Easting/Northing (m)."""
    lon0, lat0 = float(np.mean(lons)), float(np.mean(lats))
    utm_crs = utm_crs_from_lon(lon0, lat0)
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    E, N = transformer.transform(lons, lats)
    return np.column_stack((E, N)), transformer


def from_utm(E, N, transformer):
    lon, lat = transformer.transform(E, N, direction="INVERSE")
    return np.column_stack((lat, lon))  # folium expects lat, lon order


def confidence_ellipse_coords(cov: np.ndarray, center: np.ndarray, transformer, n_pts=72, alpha=0.95):
    """Return list of lat/lon pairs tracing the confidence ellipse."""
    vals, vecs = eigh(cov)
    # largest first
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    k2 = chi2.ppf(alpha, df=2)
    a, b = np.sqrt(k2 * vals)  # semi‑axes lengths (m)
    phi = np.arctan2(vecs[1, 0], vecs[0, 0])
    theta = np.linspace(0, 2 * np.pi, n_pts)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    xy = R @ np.vstack((x, y)) + center[:, None]
    latlon = from_utm(xy[0], xy[1], transformer)
    return latlon.tolist()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Bearing-only Resection", layout="wide")
st.title("Bearing-only Resection demo (ODR)")

# Session state for anchor list
def init_state():
    if "anchors" not in st.session_state:
        st.session_state.anchors = [dict(lat=0.0, lon=0.0, bearing=0.0)]

init_state()

st.sidebar.header("Anchors and bearings")

# Table‑like inputs
for idx, anc in enumerate(st.session_state.anchors):
    cols = st.sidebar.columns(3, gap="small")
    anc["lat"] = cols[0].number_input("lat°", key=f"lat{idx}", value=float(anc["lat"]))
    anc["lon"] = cols[1].number_input("lon°", key=f"lon{idx}", value=float(anc["lon"]))
    anc["bearing"] = cols[2].number_input("bearing°", key=f"brg{idx}", value=float(anc["bearing"]))

# Add/remove buttons
c1, c2 = st.sidebar.columns(2)
if c1.button("Add row"):
    st.session_state.anchors.append(dict(lat=0.0, lon=0.0, bearing=0.0))
if c2.button("Remove last") and len(st.session_state.anchors) > 1:
    st.session_state.anchors.pop()

# Noise parameters
st.sidebar.markdown("### Noise parameters")
σθ = st.sidebar.number_input("σθ (° rms compass)", value=0.5, min_value=0.01)
sigma_anchor = st.sidebar.number_input("sigma_anchor (m rms GNSS)", value=0.02, min_value=0.0)
robust = st.sidebar.checkbox("Huber robust", value=True)

run = st.sidebar.button("Solve position", type="primary")

# Prepare map placeholder
m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
TileLayer("OpenStreetMap", name="OSM").add_to(m)
TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="&copy; Esri",
    name="Satellite",
).add_to(m)
LayerControl().add_to(m)

if run:
    # Extract input arrays
    lats = np.array([a["lat"] for a in st.session_state.anchors])
    lons = np.array([a["lon"] for a in st.session_state.anchors])
    bearings_deg = np.array([a["bearing"] for a in st.session_state.anchors])

    # Projection to UTM
    anchor_xy, transformer = to_utm(lons, lats)
    sigma_theta = np.deg2rad(σθ)
    Sigma = np.repeat(np.eye(2)[None, :, :] * sigma_anchor**2, len(anchor_xy), axis=0)

    # Solve ODR
    res = solve_resection_odr(
        theta=np.deg2rad(bearings_deg),
        anchors=anchor_xy,
        sigma_theta=sigma_theta,
        Sigma=Sigma,
        robust=robust,
    )

    # Map center and display
    center_latlon = from_utm(res["position"][0], res["position"][1], transformer)[0]
    m.location = center_latlon.tolist()
    m.zoom_start = 17

    # Plot anchors
    for lat, lon, brg in zip(lats, lons, bearings_deg):
        Marker([lat, lon], popup=f"Anchor<br>{lat:.5f},{lon:.5f}<br>{brg}°").add_to(m)

    # Plot estimated position
    Marker(center_latlon, icon=folium.Icon(color="red"), popup="Estimated position").add_to(m)

    # Confidence ellipse
    ell_coords = confidence_ellipse_coords(res["cov"], res["position"], transformer)
    Polygon(ell_coords, color="red", weight=2, fill=True, fill_opacity=0.15).add_to(m)

    st.success(
        f"Position: {center_latlon[0]:.6f}°, {center_latlon[1]:.6f}°  \n"
        f"95% semi-axes: {np.sqrt(chi2.ppf(0.95,2)*np.linalg.eigvals(res['cov']).max()):.2f} m / "
        f"{np.sqrt(chi2.ppf(0.95,2)*np.linalg.eigvals(res['cov']).min()):.2f} m"
    )

st_folium(m, height=600, width=1200)
