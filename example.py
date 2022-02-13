from ImbalancedPipeline import ImbalancedPipeline
from featboost import FeatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from skrebate import ReliefF

if __name__ == '__main__':
    weights = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    estimator = XGBClassifier(eval_metric="mlogloss", n_estimators=200, max_depth=3)
    fb_classifier = FeatBoostClassifier(estimator=estimator, metric="mcc", max_number_of_features=10)
    rf_classifier = ReliefF(n_features_to_select=20)
    kNN = KNeighborsClassifier()
    fb = ImbalancedPipeline(fb_classifier, kNN, weights=weights)
    rf = ImbalancedPipeline(rf_classifier, kNN, weights=weights)
    fb.run_pipeline()
    rf.run_pipeline()