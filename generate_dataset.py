"""
CarPriceAI - Dataset Generator
Generates an authentic automotive dataset matching Kaggle's Car Details v3 schema:
name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats
"""

import csv
import math
import os
import random

# Seed for deterministic generation
random.seed(42)

CAR_PROFILES = [
    # Economy & Hatchbacks
    {"name": "Maruti Swift Dzire VDI", "brand": "Maruti", "base_mrp": 750000, "engine": 1248, "mileage": 26.59, "power": 74.0, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Maruti Alto LXi", "brand": "Maruti", "base_mrp": 380000, "engine": 796, "mileage": 22.05, "power": 47.3, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Maruti Wagon R VXI", "brand": "Maruti", "base_mrp": 550000, "engine": 998, "mileage": 21.79, "power": 67.0, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Maruti Baleno Delta 1.2", "brand": "Maruti", "base_mrp": 720000, "engine": 1197, "mileage": 21.01, "power": 81.8, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Maruti Swift VXI", "brand": "Maruti", "base_mrp": 660000, "engine": 1197, "mileage": 21.21, "power": 81.8, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Maruti Ertiga VDI", "brand": "Maruti", "base_mrp": 950000, "engine": 1248, "mileage": 24.52, "power": 88.5, "fuel": "Diesel", "trans": "Manual", "seats": 7, "segment": "Mid-Range"},
    {"name": "Hyundai i20 Magna", "brand": "Hyundai", "base_mrp": 720000, "engine": 1197, "mileage": 19.8, "power": 81.8, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Hyundai Grand i10 Sportz", "brand": "Hyundai", "base_mrp": 620000, "engine": 1197, "mileage": 18.9, "power": 81.86, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Hyundai Santro Era", "brand": "Hyundai", "base_mrp": 460000, "engine": 1086, "mileage": 20.3, "power": 68.0, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Tata Tiago XT", "brand": "Tata", "base_mrp": 580000, "engine": 1199, "mileage": 20.0, "power": 84.5, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Renault Kwid RXT", "brand": "Renault", "base_mrp": 480000, "engine": 999, "mileage": 22.3, "power": 67.0, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    
    # Mid-Range Sedans & Compact SUVs
    {"name": "Hyundai Creta 1.6 CRDi SX", "brand": "Hyundai", "base_mrp": 1400000, "engine": 1582, "mileage": 19.67, "power": 126.2, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Hyundai Creta SX Diesel Automatic", "brand": "Hyundai", "base_mrp": 1650000, "engine": 1493, "mileage": 18.5, "power": 113.4, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Mid-Range"},
    {"name": "Hyundai Verna 1.6 VTVT SX", "brand": "Hyundai", "base_mrp": 1150000, "engine": 1591, "mileage": 17.7, "power": 121.3, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Honda City i-VTEC V", "brand": "Honda", "base_mrp": 1250000, "engine": 1498, "mileage": 17.8, "power": 117.3, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Honda City ZX CVT Automatic", "brand": "Honda", "base_mrp": 1480000, "engine": 1498, "mileage": 18.0, "power": 119.3, "fuel": "Petrol", "trans": "Automatic", "seats": 5, "segment": "Mid-Range"},
    {"name": "Honda Amaze 1.2 VX i-VTEC", "brand": "Honda", "base_mrp": 820000, "engine": 1199, "mileage": 18.6, "power": 88.5, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Kia Seltos HTX D", "brand": "Kia", "base_mrp": 1550000, "engine": 1493, "mileage": 20.8, "power": 113.4, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Kia Seltos GTX Plus AT", "brand": "Kia", "base_mrp": 1850000, "engine": 1353, "mileage": 16.5, "power": 138.0, "fuel": "Petrol", "trans": "Automatic", "seats": 5, "segment": "Mid-Range"},
    {"name": "Tata Nexon 1.5 Revotorq XZ Plus", "brand": "Tata", "base_mrp": 1100000, "engine": 1497, "mileage": 21.5, "power": 108.5, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Tata Harrier XZ", "brand": "Tata", "base_mrp": 1850000, "engine": 1956, "mileage": 16.35, "power": 167.6, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Mahindra XUV500 W8 2WD", "brand": "Mahindra", "base_mrp": 1550000, "engine": 2179, "mileage": 15.1, "power": 140.0, "fuel": "Diesel", "trans": "Manual", "seats": 7, "segment": "Mid-Range"},
    {"name": "Mahindra Scorpio S11", "brand": "Mahindra", "base_mrp": 1680000, "engine": 2179, "mileage": 15.0, "power": 140.0, "fuel": "Diesel", "trans": "Manual", "seats": 7, "segment": "Mid-Range"},
    {"name": "Mahindra Thar LX 4-Str Hard Top AT", "brand": "Mahindra", "base_mrp": 1650000, "engine": 1997, "mileage": 15.2, "power": 150.0, "fuel": "Petrol", "trans": "Automatic", "seats": 4, "segment": "Mid-Range"},
    {"name": "Ford EcoSport 1.5 TDCi Titanium", "brand": "Ford", "base_mrp": 1050000, "engine": 1498, "mileage": 21.7, "power": 98.6, "fuel": "Diesel", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Ford Endeavour 2.0 Titanium Plus 4x4 AT", "brand": "Ford", "base_mrp": 3400000, "engine": 1996, "mileage": 12.4, "power": 167.6, "fuel": "Diesel", "trans": "Automatic", "seats": 7, "segment": "Luxury"},
    {"name": "Toyota Innova Crysta 2.4 VX", "brand": "Toyota", "base_mrp": 2100000, "engine": 2393, "mileage": 15.1, "power": 147.5, "fuel": "Diesel", "trans": "Manual", "seats": 7, "segment": "Mid-Range"},
    {"name": "Toyota Fortuner 2.8 4x4 AT", "brand": "Toyota", "base_mrp": 3800000, "engine": 2755, "mileage": 14.2, "power": 201.1, "fuel": "Diesel", "trans": "Automatic", "seats": 7, "segment": "Luxury"},
    {"name": "Toyota Glanza G", "brand": "Toyota", "base_mrp": 840000, "engine": 1197, "mileage": 22.35, "power": 88.5, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Economy"},
    {"name": "Volkswagen Polo 1.0 TSI Highline Plus", "brand": "Volkswagen", "base_mrp": 920000, "engine": 999, "mileage": 18.24, "power": 108.6, "fuel": "Petrol", "trans": "Manual", "seats": 5, "segment": "Mid-Range"},
    {"name": "Skoda Octavia 2.0 TDI Style AT", "brand": "Skoda", "base_mrp": 2400000, "engine": 1968, "mileage": 19.3, "power": 141.0, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Mid-Range"},

    # Premium & Luxury Vehicles
    {"name": "BMW 3 Series 320d Luxury Line", "brand": "BMW", "base_mrp": 4800000, "engine": 1995, "mileage": 19.62, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "BMW 5 Series 520d Luxury Line", "brand": "BMW", "base_mrp": 6400000, "engine": 1995, "mileage": 17.42, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "BMW X1 sDrive20d xLine", "brand": "BMW", "base_mrp": 4200000, "engine": 1995, "mileage": 19.62, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "Mercedes-Benz C-Class C 220d", "brand": "Mercedes-Benz", "base_mrp": 5400000, "engine": 1950, "mileage": 17.9, "power": 191.8, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "Mercedes-Benz E-Class Exclusive E 220 d", "brand": "Mercedes-Benz", "base_mrp": 7200000, "engine": 1950, "mileage": 16.1, "power": 191.8, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "Audi A4 35 TDI Premium Plus", "brand": "Audi", "base_mrp": 4400000, "engine": 1968, "mileage": 17.84, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "Audi A6 35 TDI Matrix", "brand": "Audi", "base_mrp": 5800000, "engine": 1968, "mileage": 16.66, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
    {"name": "Audi Q5 40 TDI quattro Technology", "brand": "Audi", "base_mrp": 6200000, "engine": 1968, "mileage": 15.3, "power": 187.7, "fuel": "Diesel", "trans": "Automatic", "seats": 5, "segment": "Luxury"},
]

OWNER_TYPES = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]

def generate_records(target_count=4200):
    records = []
    
    for i in range(target_count):
        profile = random.choice(CAR_PROFILES)
        
        # Year distribution: 2008 to 2024
        year_weights = [1, 2, 3, 5, 8, 12, 16, 20, 22, 24, 25, 26, 22, 18, 12, 6, 2]
        years = list(range(2008, 2025))
        year = random.choices(years, weights=year_weights)[0]
        
        car_age = 2026 - year
        
        # Kilometers driven logic: roughly 8k to 22k km per year
        avg_km_per_year = random.uniform(8000, 19000)
        km_driven = int(avg_km_per_year * car_age + random.gauss(0, 7000))
        km_driven = max(4000, min(km_driven, 280000))
        
        # Fuel type variation
        fuel = profile["fuel"]
        if random.random() < 0.08 and fuel == "Petrol":
            fuel = "CNG"
        elif random.random() < 0.02 and fuel == "Petrol":
            fuel = "LPG"
            
        # Transmission variation
        trans = profile["trans"]
        if random.random() < 0.12 and trans == "Manual" and profile["segment"] != "Economy":
            trans = "Automatic"
            
        # Owner selection based on age
        if car_age <= 2:
            owner = random.choices(OWNER_TYPES, weights=[90, 8, 2, 0])[0]
        elif car_age <= 5:
            owner = random.choices(OWNER_TYPES, weights=[65, 28, 6, 1])[0]
        elif car_age <= 9:
            owner = random.choices(OWNER_TYPES, weights=[40, 42, 14, 4])[0]
        else:
            owner = random.choices(OWNER_TYPES, weights=[20, 45, 25, 10])[0]
            
        seller_type = random.choices(SELLER_TYPES, weights=[68, 26, 6])[0]
        
        # Engine, mileage, power variations around base profile
        engine_cc = profile["engine"] + random.choice([0, 0, 0, -50, 50])
        mileage_val = round(profile["mileage"] + random.gauss(0, 1.2), 2)
        power_val = round(profile["power"] + random.gauss(0, 2.5), 1)
        seats = profile["seats"]
        
        # Realistic Automotive Resale Price Formula (Industrial Calibration)
        base_mrp = profile["base_mrp"]
        
        # Depreciation curve: exponential decay
        depreciation_rate = 0.88 ** car_age
        
        # KM penalty: ~1.6% per 10,000 km
        km_penalty = max(0.65, 1.0 - (km_driven / 100000) * 0.16)
        
        # Owner penalty
        owner_multipliers = {
            "First Owner": 1.0,
            "Second Owner": 0.88,
            "Third Owner": 0.77,
            "Fourth & Above Owner": 0.68
        }
        owner_factor = owner_multipliers[owner]
        
        # Seller type adjustment
        seller_multipliers = {
            "Individual": 0.96,
            "Dealer": 1.04,
            "Trustmark Dealer": 1.10
        }
        seller_factor = seller_multipliers[seller_type]
        
        # Transmission premium
        trans_factor = 1.12 if trans == "Automatic" else 1.0
        
        # Fuel premium
        fuel_factors = {"Diesel": 1.05, "Petrol": 1.0, "CNG": 0.94, "LPG": 0.88, "Electric": 1.08}
        fuel_factor = fuel_factors.get(fuel, 1.0)
        
        # Random market noise (±5%)
        noise = random.uniform(0.95, 1.05)
        
        calculated_price = base_mrp * depreciation_rate * km_penalty * owner_factor * seller_factor * trans_factor * fuel_factor * noise
        
        # Boundary constraints
        selling_price = int(max(45000, round(calculated_price / 1000) * 1000))
        
        # Formatting with units as in Kaggle Car Details v3
        mileage_str = f"{mileage_val} kmpl" if fuel != "CNG" else f"{mileage_val} km/kg"
        engine_str = f"{engine_cc} CC"
        power_str = f"{power_val} bhp"
        torque_str = f"{int(power_val * 1.8)}Nm@ 2000rpm"
        
        # Introduce a few realistic missing values (~1.5%) to exercise data wrangling imputation
        if random.random() < 0.015:
            mileage_str = ""
        if random.random() < 0.012:
            engine_str = ""
        if random.random() < 0.015:
            power_str = ""
        if random.random() < 0.01:
            seats = ""

        record = {
            "name": profile["name"],
            "year": year,
            "selling_price": selling_price,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": trans,
            "owner": owner,
            "mileage": mileage_str,
            "engine": engine_str,
            "max_power": power_str,
            "torque": torque_str,
            "seats": seats
        }
        records.append(record)
        
    # Inject a few duplicate rows to exercise duplicate removal in Module 3
    for _ in range(40):
        records.append(dict(random.choice(records)))
        
    return records

def save_to_csv(filepath="data/car_details_v3.csv"):
    records = generate_records(4200)
    fieldnames = [
        "name", "year", "selling_price", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} records saved to {filepath}")

if __name__ == "__main__":
    save_to_csv()
