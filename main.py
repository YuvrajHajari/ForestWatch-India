import os
import cv2
import numpy as np
import shutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ForestWatch API")

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# 2. CORS - Allow your Netlify Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://forestwatch-india.netlify.app",
        "http://localhost:5500",  # Keep for local testing
        "http://127.0.0.1:5500"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CITIES = {
    "bengaluru": {"label": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "chennai":   {"label": "Chennai",   "lat": 13.0827, "lon": 80.2707},
}

FALLBACK_PIXEL_HA = 0.0001

# HELPER: Dynamically get the base URL (Handles Local vs Render)
def get_base_url(request: Request):
    return str(request.base_url).rstrip("/") + "/static"

def compute_vari_mask(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (512, 512)).astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    vari = (G - R) / (G + R - B + 1e-6)
    brightness = (R + G + B) / 3.0
    veg = ((vari > 0.08) & (brightness > 30)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, kernel)
    return veg

def carbon_metrics(mask: np.ndarray) -> dict:
    forest_px = int(np.sum(mask > 127))
    total_px = mask.size
    area_ha = round(forest_px * FALLBACK_PIXEL_HA, 2)
    carbon_t = round(area_ha * 190.0, 2)
    co2_t = round(carbon_t * 3.67, 2)
    pct = round((forest_px / total_px) * 100, 1)
    return {"forest_area_ha": area_ha, "carbon_t": carbon_t, "co2_equiv_t": co2_t, "coverage_pct": pct}

def make_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (512, 512))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    fg = mask > 127
    out[fg] = (bgr[fg] * 0.45 + np.array([0, 200, 80], np.float32) * 0.55).astype(np.uint8)
    return out

def load_satellite(city: str, year: str) -> np.ndarray:
    path = os.path.join(STATIC_DIR, f"satellite_{city}_{year}.png")
    if not os.path.exists(path):
        path = os.path.join(STATIC_DIR, f"satellite_{year}.png")
    if not os.path.exists(path):
        raise HTTPException(404, f"No satellite image for {city} {year}.")
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.get("/cities")
def list_cities():
    return {"cities": [{"key": k, "label": v["label"]} for k, v in CITIES.items()]}

@app.get("/analyze/{city}/{year}")
def analyze(city: str, year: str, request: Request):
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_rgb = load_satellite(city, year)
    mask = compute_vari_mask(img_rgb)
    metrics = carbon_metrics(mask)
    overlay = make_overlay(img_rgb, mask)

    mask_path = os.path.join(STATIC_DIR, f"mask_{city}_{year}.png")
    overlay_path = os.path.join(STATIC_DIR, f"overlay_{city}_{year}.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    base_static = get_base_url(request)
    return {
        "city": city, "year": year,
        "satellite_url": f"{base_static}/satellite_{city}_{year}.png",
        "mask_url":      f"{base_static}/mask_{city}_{year}.png",
        "overlay_url":   f"{base_static}/overlay_{city}_{year}.png",
        **metrics,
    }

@app.get("/compare/{city}/{year_a}/{year_b}")
def compare(city: str, year_a: str, year_b: str, request: Request):
    img_a = load_satellite(city, year_a)
    img_b = load_satellite(city, year_b)
    mask_a = compute_vari_mask(img_a)
    mask_b = compute_vari_mask(img_b)

    forest_a = mask_a > 127
    forest_b = mask_b > 127
    lost = forest_a & ~forest_b
    gained = ~forest_a & forest_b
    stable = forest_a & forest_b

    diff_img = np.zeros((512, 512, 3), dtype=np.uint8)
    diff_img[stable] = [0, 200, 80]
    diff_img[lost]   = [60, 60, 220]
    diff_img[gained] = [220, 180, 0]

    diff_path = os.path.join(STATIC_DIR, f"diff_{city}_{year_a}_{year_b}.png")
    cv2.imwrite(diff_path, diff_img)

    base_static = get_base_url(request)
    return {
        "diff_url": f"{base_static}/diff_{city}_{year_a}_{year_b}.png",
        "year_a_data": carbon_metrics(mask_a),
        "year_b_data": carbon_metrics(mask_b),
        "change": {
            "lost_ha": round(int(np.sum(lost)) * FALLBACK_PIXEL_HA, 2),
            "gained_ha": round(int(np.sum(gained)) * FALLBACK_PIXEL_HA, 2),
            "net_ha": round((int(np.sum(gained)) - int(np.sum(lost))) * FALLBACK_PIXEL_HA, 2),
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT as an environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)