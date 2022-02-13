import numpy as np
from featboost import FeatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ImbalancedPipeline:
    """
       Imbalanced pipeline to run different pipelines
       for different feature selection algorithms, classifiers
       and datasets.

    Parameters
    ----------
        feature_selector : object, required
                A feature selection algorithm, with a 'fit' method
                that returns the selected subset of that run.
                Feature selection methods supported:
                1) FeatBoost
                2) ReliefF

        classifier : object, required
                A sklearn classifier, with 'fit', 'score', 'predict_proba'
                and 'predict' methods.

        dataset: string, Optional (default = 'Madelon')
                A dataset name currently supported datasets
                1) Madelon

        stability_measure: string, Optional (default = 'novel')
                A string defining the stability measure used in the pipeline.
                Currently only the 'novel' stability measure is available.

        weights : list of decimals, Optional 
            (default=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
                Specifies the percentages of the imbalance. i.e., 0.5 = 50/50, 0.4=60/40 

        iterations: int, Optional (default = 3)
            Nr. of bootstrap iterations per weight.

        Attributes
    ----------
        stability: float
            Returns the stability from the pipeline.
        aucs: array, shape = [len(weights), iterations]
            Returns the array of area under curve scores per weight.
        mAPs: array, shape = [len(weights)]
            Returns the mean average precision score per weight
        
    """

    def __init__(
        self, 
        feature_selector, 
        classifier, 
        weights,
        dataset="madelon",
        stability_measure="novel", 
        iterations=3
    ):
        self.feature_selector = feature_selector
        self.classifier = classifier
        self.stability_measure=stability_measure
        self.weights=weights
        self.iterations = iterations
        self.dataset = dataset
    
    def run_pipeline(self):
        """
        Runs the pipeline with the selected feature selection algorithm,
        classifier and dataset
        Returns
        -------
        self : object
        """
        return self._run_pipeline()

    def _run_pipeline(self):
        """
        Performs the pipeline per weight for a number of iterations,
        The subset, score, recall, precision, auc, mAP and stability
        are calculated based on the user specified classifier and feature selector.
        """
        self.selected_subsets = []
        self.aucs = []
        self.mAPs = []
        random_state = np.random.randint(1000, size=3)
        for weight in self.weights:
            self.fpr = []
            self.tpr = []
            self.auc = []
            self.recall_curve = []
            self.average_precision = []
            for i in range(self.iterations):
                X, y = make_classification(
                    n_samples=1000, random_state=random_state[i], n_redundant=0, n_informative=2, shuffle=False, flip_y=0.0, weights=[weight]
                )
                self.dimensions = X.shape[1]
                subset, score, recall, precision = self._get_score(X, y)
                print(f"\n Information at iteration {i} of {weight}")
                print(f'\t Subset: {subset}')
                print(f'\t Score: {score}')
                print(f'\t Recall: {recall}')
                print(f'\t Precision: {precision}\n')
            self._generate_plots(weight)
        self.stability = self._calculate_stability(self.selected_subsets)
        return

    def _calculate_stability(self, subsets):
        """
        Determines the stability of the subsets based on the
        selected subsets of "_run_pipeline"
        """
        d = self.dimensions
        M = len(self.weights)*3
        Z = np.zeros((M,d))
        for x, arr in enumerate(subsets):
            for item in arr:
                Z[x][item] = 1
        hatPF = np.mean(Z, axis=0)
        kbar=np.sum(hatPF)
        denom = (kbar/d)*(1-kbar/d)
        stability = 1-(M/(M-1))*np.mean(np.multiply(hatPF, 1-hatPF))/denom
        return stability

    def _get_score(self, X, y):
        """
        Determines the subset, score, recall and precision from
        the feature subset based on the feature selection algorithm
        by the user.
        """
        if isinstance(self.feature_selector, FeatBoostClassifier):
            self.feature_selector.fit(X, y)
            subset = self.feature_selector.selected_subset_
            self.selected_subsets.append(subset)
            X_new, y_new = make_classification(n_samples=500, random_state=0, n_redundant=0, n_informative=2, shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=400, train_size=100, random_state=0, shuffle=None)
            X_train = X_train[:,[x for x in subset]]
            X_test = X_test[:, [x for x in subset]]
            self.classifier.fit(X_train, y_train)
            score = self.classifier.score(X_test, y_test)
            predicted = self.classifier.predict(X_test)
            y_predict = self.classifier.predict_proba(X_test)[:, 1]
            precision_list, recall_list, _ = precision_recall_curve(y_test, y_predict)
            auc = roc_auc_score(y_test, y_predict)
            fpr, tpr, _ = roc_curve(y_test, y_predict)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.auc.append(auc)
            self.recall_curve.append((recall_list, precision_list))
            self.average_precision.append(average_precision_score(y_test, y_predict))
            _, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()
            recall = TP/(TP+FP)
            precision = TP/(TP+FN)
            return subset, score, recall, precision
        elif isinstance(self.feature_selector, ReliefF):
            self.feature_selector.fit(X, y)
            Relief_score = self.feature_selector.feature_importances_
            temp = [i for i in Relief_score if i>np.mean(Relief_score)]
            subset = np.argsort(-Relief_score)[:len(temp)]
            self.selected_subsets.append(subset)
            X_new, y_new = make_classification(n_samples=500, random_state=0, n_redundant=0, n_informative=2, shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=400, train_size=100, random_state=0, shuffle=None)
            X_train = X_train[:,[x for x in subset]]
            X_test = X_test[:, [x for x in subset]]
            self.classifier.fit(X_train, y_train)
            score = self.classifier.score(X_test, y_test)
            predicted = self.classifier.predict(X_test)
            y_predict = self.classifier.predict_proba(X_test)[:, 1]
            precision_list, recall_list, _ = precision_recall_curve(y_test, y_predict)
            auc = roc_auc_score(y_test, y_predict)
            fpr, tpr, _ = roc_curve(y_test, y_predict)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.auc.append(auc)
            self.recall_curve.append((recall_list, precision_list))
            self.average_precision.append(average_precision_score(y_test, y_predict))
            _, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()
            recall = TP/(TP+FP)
            precision = TP/(TP+FN)
            return subset, score, recall, precision 

    def _generate_plots(self, weight):
        """
        Generates ROC- and PR curves for the particular weight. 
        """
        self.name = self.feature_selector.__class__.__name__
        colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']
        mean_auc = np.mean(self.auc)
        std_auc = np.std(self.auc)

        self.aucs.append(self.auc)
        mAP = np.sum(self.average_precision) / self.iterations
        std_mAP = np.std(self.average_precision)
        self.mAPs.append(mAP)

        for i in range(self.iterations):
            plt.figure(1)
            plt.plot(self.recall_curve[i][0], self.recall_curve[i][1], color=colors[i], label=f'Iteration {i+1}')
            plt.title(self.name + ' ' + r'(mAP = %0.2f $\pm$ %0.2f - Weight = %0.2f)' % (mAP, std_mAP, weight))
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.legend(loc="lower right")


            plt.figure(2)
            plt.plot(self.fpr[i], self.tpr[i], color=colors[i], label=f'Iteration {i+1}')
            plt.title(self.name + ' ' + r'(AUC = %0.2f $\pm$ %0.2f - Weight = %0.2f)' % (mean_auc, std_auc, weight) )
            plt.ylabel('True positive rate')
            plt.xlabel('False positive rate')
            plt.legend(loc="lower right")
        plt.figure(1)
        plt.savefig(f'figures/{self.name}_pr_curve_{weight}_.png')
        plt.clf()
        plt.figure(2)
        plt.savefig(f'figures/{self.name}_rocauc_{weight}_.png')
        plt.clf()