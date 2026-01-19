import pandas as pd
import json
from scipy.stats import describe
import numpy as np
from stat_functions import analyze_array, goal_maker

d = dict()

with open("tht_bazaar.json") as json_data:
    d = json.load(json_data)
    json_data.close()
    # print(d)

d = d.get("entries")

data = pd.DataFrame(d)
print(f"Total Players in Legendary: {len(data)}")
# print(data.head)

shanes_rating = data[ data["Username"] == "ZombieReaper" ]["Rating"].to_list()[0]
user = "Death913"

# Death913
user_rating = data[ data["Username"] == user ]["Rating"].to_list()[0]
rating_diff = user_rating - shanes_rating
print(f"{user} Rating is {user_rating} which is {abs(rating_diff)} { 'Greater' if rating_diff > 0 else 'Less Than' } then mine.")

# print("Columns")
# print(data.columns)

ratings = data["Rating"].to_list()
# print(ratings)

analyze_array(ratings)
goal_maker( shanes_rating, ratings )