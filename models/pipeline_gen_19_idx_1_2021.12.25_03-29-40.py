import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.6514823370702081
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=20.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.0001)),
    StackingEstimator(estimator=LinearSVC(C=0.5, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
    XGBClassifier(learning_rate=0.01, max_depth=7, min_child_weight=5, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
