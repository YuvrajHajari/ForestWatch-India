import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ForestWatch API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CITIES = {
    "bengaluru": {"label": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "chennai":   {"label": "Chennai",   "lat": 13.0827, "lon": 80.2707},
}

FALLBACK_PIXEL_HA = 0.0001
BASE_URL = "http://127.0.0.1:8000/static"



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
        raise HTTPException(404, f"No satellite image for {city} {year}. Run satellite_fetch.py first.")
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



@app.get("/cities")
def list_cities():
    return {"cities": [{"key": k, "label": v["label"]} for k, v in CITIES.items()]}


@app.get("/years/{city}")
def list_years(city: str):
    years = set()
    for f in os.listdir(STATIC_DIR):
        if f.startswith(f"satellite_{city}_") and f.endswith(".png"):
            y = f.replace(f"satellite_{city}_", "").replace(".png", "")
            years.add(y)
    return {"years": sorted(years)}


@app.get("/analyze/{city}/{year}")
def analyze(city: str, year: str):
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_rgb = load_satellite(city, year)
    mask = compute_vari_mask(img_rgb)
    metrics = carbon_metrics(mask)
    overlay = make_overlay(img_rgb, mask)

    mask_path = os.path.join(STATIC_DIR, f"mask_{city}_{year}.png")
    overlay_path = os.path.join(STATIC_DIR, f"overlay_{city}_{year}.png")
    sat_path_std = os.path.join(STATIC_DIR, f"satellite_{city}_{year}.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    if not os.path.exists(sat_path_std):
        old = os.path.join(STATIC_DIR, f"satellite_{year}.png")
        if os.path.exists(old):
            import shutil; shutil.copy(old, sat_path_std)

    return {
        "city": city, "year": year,
        "satellite_url": f"{BASE_URL}/satellite_{city}_{year}.png",
        "mask_url":      f"{BASE_URL}/mask_{city}_{year}.png",
        "overlay_url":   f"{BASE_URL}/overlay_{city}_{year}.png",
        **metrics,
    }


@app.get("/compare/{city}/{year_a}/{year_b}")
def compare(city: str, year_a: str, year_b: str):
    
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_a = load_satellite(city, year_a)
    img_b = load_satellite(city, year_b)
    mask_a = compute_vari_mask(img_a)
    mask_b = compute_vari_mask(img_b)

    forest_a = mask_a > 127
    forest_b = mask_b > 127

    lost    = forest_a & ~forest_b   
    gained  = ~forest_a & forest_b   
    stable  = forest_a & forest_b    

    diff_img = np.zeros((512, 512, 3), dtype=np.uint8)
    diff_img[stable] = [0, 200, 80]    
    diff_img[lost]   = [60, 60, 220]   
    diff_img[gained] = [220, 180, 0]   

    diff_path = os.path.join(STATIC_DIR, f"diff_{city}_{year_a}_{year_b}.png")
    cv2.imwrite(diff_path, diff_img)

    metrics_a = carbon_metrics(mask_a)
    metrics_b = carbon_metrics(mask_b)

    lost_ha   = round(int(np.sum(lost))   * FALLBACK_PIXEL_HA, 2)
    gained_ha = round(int(np.sum(gained)) * FALLBACK_PIXEL_HA, 2)
    net_ha    = round(gained_ha - lost_ha, 2)

    return {
        "city": city,
        "year_a": year_a, "year_b": year_b,
        "diff_url": f"{BASE_URL}/diff_{city}_{year_a}_{year_b}.png",
        "year_a_data": metrics_a,
        "year_b_data": metrics_b,
        "change": {
            "lost_ha":   lost_ha,
            "gained_ha": gained_ha,
            "net_ha":    net_ha,
            "lost_carbon_t":  round(lost_ha * 190.0, 2),
            "lost_co2_t":     round(lost_ha * 190.0 * 3.67, 2),
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://forestwatch-india.netlify.app"], # Your Netlify link
    allow_methods=["*"],
    allow_headers=["*"],
)