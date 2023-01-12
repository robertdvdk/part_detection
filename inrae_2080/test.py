"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import pandas as pd
# Function definitions
def main():
    df = pd.DataFrame([1, 2, 3, 4, 5])
    print(df)
    print(df.iloc[[2, 3, 4]])

if __name__ == "__main__":
    main()
