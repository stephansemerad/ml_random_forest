from mlrf import ml
from rich import print

ml = ml()

ml.sample_data = {
    "city": "Berlin",
    "country_code": "DE",
    "district": "Charlottenburg",
    "area_sqm": 75,
    "price_eur": 300000,
    "rooms": 3,
    "bathrooms": 2,
    "balcony": True,
    "airconditioning": True,
    "garden": False,
    "parking": True,
    "furnished": True,
    "offer_type": "sale",
}


ml.load_csv("data.csv")

ml.create_rf_pipeline()
ml.make_rf_prediction()
print()
ml.create_xgb_pipeline()
ml.make_xgb_prediction()

print()
ml.threshold = 0.05
ml.calculate_accuracy()


print()
ml.threshold = 0.025
ml.calculate_accuracy()


print()
ml.threshold = 0.01
ml.calculate_accuracy()
