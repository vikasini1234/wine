# test_model.py
import unittest
import xmlrunner
import pandas as pd
import numpy as np
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
    
    def test_missing_value_handling(self):
        # TC2: Missing Value Handling
        # Create a copy of X_test and introduce NaN values in the 'fixed acidity' column
        X_test_with_missing = X_test.copy()
        X_test_with_missing.loc[0, 'fixed acidity'] = np.nan
        # We expect the model to raise a ValueError when predicting with NaN values
        with self.assertRaises(ValueError):
            regr.predict(X_test_with_missing)

# Run the tests
if __name__ == '__main__':
    with open('test-reports/results.xml', 'wb') as output:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output='test-reports'),
            failfast=False, buffer=False, catchbreak=False)
