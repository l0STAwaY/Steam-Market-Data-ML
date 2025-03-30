import pandas as pd

# Example string with a prefix
date_string = "May 12, 2024"



# Convert the cleaned string to a datetime object
date = pd.to_datetime(date_string)

print(date)  # Output: 2024-05-12 00:00:00