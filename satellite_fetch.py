
import ee, requests, numpy as np, cv2, os

GEE_PROJECT = "insights-ba743"

CITIES = {
    "chennai": {"lat": 13.0827, "lon": 80.2707, "buffer_km": 6},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "buffer_km": 6},
}

YEARS = [2016, 2019, 2022, 2025]
IMAGE_SIZE = 512
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def init():
    ee.Initialize(project=GEE_PROJECT)
    print("✅ GEE initialized")


def fetch_image(city_key, cfg, year):
    point  = ee.Geometry.Point([cfg["lon"], cfg["lat"]])
    region = point.buffer(cfg["buffer_km"] * 1000)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B4", "B3", "B2"])
        .median()
    )
    url = collection.getThumbURL({
        "region": region, "dimensions": IMAGE_SIZE,
        "format": "png", "min": 0, "max": 3000, "gamma": 1.4,
    })
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_all():
    init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for city_key, cfg in CITIES.items():
        for year in YEARS:
            out = os.path.join(OUTPUT_DIR, f"satellite_{city_key}_{year}.png")
            if os.path.exists(out):
                print(f"  skip {city_key} {year}")
                continue
            print(f"📡 {city_key} {year}...")
            try:
                img = fetch_image(city_key, cfg, year)
                cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"  ✅ saved")
            except Exception as e:
                print(f"  ❌ {e}")


if __name__ == "__main__":
    fetch_all()