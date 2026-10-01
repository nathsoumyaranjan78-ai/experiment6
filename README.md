# experiment6
import numpy as np

# Harvest data
# Rows = Tomatoes, Carrots, Potatoes, Onions
# Columns = Monday, Tuesday, Wednesday, Thursday,
#           Friday, Saturday, Sunday

harvest_data = np.array([
    [50, 55, 60, 52, 58, 65, 70],       # Tomatoes
    [30, 32, 28, 35, 33, 40, 42],       # Carrots
    [100, 95, 105, 110, 98, 120, 125],  # Potatoes
    [40, 38, 42, 45, 41, 50, 55]        # Onions
])

# Crop names and days
crops = ["Tomatoes", "Carrots", "Potatoes", "Onions"]

days = ["Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"]


# --------------------------------------------------
# Q1. Array Properties
# --------------------------------------------------

print("Q1. Array Properties")

print("Shape:", harvest_data.shape)
print("Number of dimensions:", harvest_data.ndim)
print("Number of elements:", harvest_data.size)
print("Data type:", harvest_data.dtype)


# --------------------------------------------------
# Q2. Restructuring the array
# Rows = days
# Columns = crops
# --------------------------------------------------

print("\nQ2. Restructured Array")

restructured_data = harvest_data.reshape(4, 7).T

print(restructured_data)

print("New Shape:", restructured_data.shape)


# --------------------------------------------------
# Q3. Weekly Crop Yield
# Total yield for each crop
# --------------------------------------------------

print("\nQ3. Weekly Crop Yield")

weekly_yield = np.sum(harvest_data, axis=1)

for i in range(len(crops)):
    print(crops[i], "=", weekly_yield[i], "kg")


# --------------------------------------------------
# Q4. Daily Average
# Average yield of all crops for each day
# --------------------------------------------------

print("\nQ4. Daily Average")

daily_average = np.mean(harvest_data, axis=0)

for i in range(len(days)):
    print(days[i], "=", daily_average[i], "kg")


# --------------------------------------------------
# Q5. Best Day
# Maximum yield produced by any crop on any day
# --------------------------------------------------

print("\nQ5. Best Day")

maximum_yield = np.max(harvest_data)

position = np.unravel_index(
    np.argmax(harvest_data),
    harvest_data.shape
)

crop_index = position[0]
day_index = position[1]

print("Maximum Yield:", maximum_yield, "kg")
print("Crop:", crops[crop_index])
print("Day:", days[day_index])


# --------------------------------------------------
# Q6. Broadcasting
# Fertilizer Bonus
# --------------------------------------------------

print("\nQ6. Fertilizer Bonus")

# Extra kg for each day
daily_bonus = np.array([0, 0, 0, 5, 10, 15, 20])

# Add bonus to every crop using broadcasting
bonus_data = harvest_data + daily_bonus

print("After Fertilizer Bonus:")
print(bonus_data)


# --------------------------------------------------
# Shrinkage after transportation
# --------------------------------------------------

# Shrinkage factors for each crop
shrinkage_factor = np.array([0.95, 0.98, 0.97, 0.99])

# Reshape to (4,1) for broadcasting
shrinkage_factor = shrinkage_factor.reshape(4, 1)

# Apply shrinkage
final_harvest = bonus_data * shrinkage_factor

print("\nFinal Transported Harvest:")
print(final_harvest)
