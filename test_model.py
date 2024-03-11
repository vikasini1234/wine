# test_model.py
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess the dataset
df = pd.read_csv('wine_quality.csv')
# Ensure 'total sulfur dioxide' is of type int, if it's supposed to be an integer.
df['total sulfur dioxide'] = df['total sulfur dioxide'].astype(int)
X = df.drop('quality', axis=1)
y = df['quality']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
regr = RandomForestRegressor(max_depth=5, random_state=42)
regr.fit(X_train, y_train)

# Test cases
class TestWineModel(unittest.TestCase):

    def test_data_format_validation(self):
        # Ensure 'pH' is of float type
        self.assertTrue(pd.api.types.is_float_dtype(X_train['pH']), "pH values must be float")
        # Ensure 'total sulfur dioxide' is of integer type after conversion
        self.assertTrue(pd.api.types.is_integer_dtype(X_train['total sulfur dioxide']), "Total sulfur dioxide values must be integer")

# Run the tests
if __name__ == '__main__':
    unittest.main()
