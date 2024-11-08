from imblearn.ensemble import RUSBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


MODEL_MAPPING_HR = {
    "sq": {
        "ecg": {
            "random_forests": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=10,
                min_samples_leaf=2,
                max_features="sqrt",
                max_depth=20,
                bootstrap=True,
            ),
            "rusboosted_decision_trees": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(
                    splitter="random",
                    min_samples_split=2,
                    min_samples_leaf=2,
                    min_impurity_decrease=0,
                    max_leaf_nodes=50,
                    max_features="log2",
                ),
                n_estimators=60,
                learning_rate=0.1,
            ),
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    max_depth=20,
                    bootstrap=False,
                ),
                n_estimators=40,
                learning_rate=0.1,
            ),
        },
        "ppg": {
            "knn": KNeighborsClassifier(
                weights="uniform", p=2, n_neighbors=1, algorithm="auto"
            ),
            "logistic_regression": LogisticRegression(
                solver="saga", penalty="l2", max_iter=100, C=0.033598
            ),
            "gaussian_naive_bayes": GaussianNB(var_smoothing=0.1),
            "decision_trees": DecisionTreeClassifier(
                splitter="best",
                min_samples_split=10,
                min_samples_leaf=1,
                min_impurity_decrease=0.0,
                max_leaf_nodes=50,
                max_features=None,
                max_depth=50,
                criterion="gini",
                class_weight=None,
            ),
            "random_forests": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features="sqrt",
                max_depth=30,
                bootstrap=True,
            ),
            "gradient_boosting_machines": GradientBoostingClassifier(
                subsample=0.8,
                n_estimators=300,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features="log2",
                max_depth=3,
                learning_rate=0.1,
            ),
            "rusboosted_decision_trees": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(
                    splitter="random",
                    min_samples_split=5,
                    min_samples_leaf=4,
                    min_impurity_decrease=0.0,
                    max_leaf_nodes=None,
                    max_features="sqrt",
                    max_depth=20,
                    criterion="gini",
                    class_weight=None,
                ),
                n_estimators=40,
                learning_rate=0.1,
            ),
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="log2",
                    max_depth=20,
                    bootstrap=False,
                ),
                n_estimators=30,
                learning_rate=0.1,
            ),
        },
    },
    "ma": {
        "ecg": {
            "rusboosted_decision_trees": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(
                    splitter="random",
                    min_samples_split=2,
                    min_samples_leaf=2,
                    min_impurity_decrease=0,
                    max_leaf_nodes=50,
                    max_features="log2",
                ),
                n_estimators=60,
                learning_rate=10,
            )
        },
        "ppg": {
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    max_depth=10,
                    bootstrap=False,
                ),
                n_estimators=40,
                learning_rate=0.1,
            )
        },
    },
}

MODEL_MAPPING_RR = {
    "sq": {
        "scg": {
            "random_forests": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features="sqrt",
                max_depth=30,
                bootstrap=True,
            ),
            "random_forests_balanced": RandomForestClassifier(
                n_estimators=40,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=30,
                bootstrap=True,
                class_weight="balanced",
            ),
            "random_forests_balanced_0_oriented": RandomForestClassifier(
                n_estimators=40,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="log2",
                max_depth=10,
                bootstrap=True,
                class_weight="balanced",
            ),
            "random_forests_balanced_subsample": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=None,
                bootstrap=True,
                class_weight="balanced_subsample",
            ),
            "random_forests_balanced_subsample_0_oriented": RandomForestClassifier(
                n_estimators=30,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features="sqrt",
                max_depth=10,
                bootstrap=True,
                class_weight="balanced_subsample",
            ),
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    max_depth=10,
                    bootstrap=False,
                ),
                n_estimators=40,
                learning_rate=0.1,
            ),
        },
        "mi": {
            "gaussian_naive_bayes": GaussianNB(var_smoothing=0.0001),
            "rusboosted_decision_trees": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(
                    splitter="random",
                    min_samples_split=10,
                    min_samples_leaf=1,
                    min_impurity_decrease=0.0,
                    max_leaf_nodes=None,
                    max_features="sqrt",
                    max_depth=20,
                    criterion="gini",
                    class_weight=None,
                ),
                n_estimators=60,
                learning_rate=40,
            ),
            "random_forests_balanced": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="log2",
                max_depth=30,
                bootstrap=True,
                class_weight="balanced",
            ),
            "random_forests_balanced_0": RandomForestClassifier(
                n_estimators=50,
                min_samples_split=10,
                min_samples_leaf=10,
                max_features="log2",
                max_depth=10,
                bootstrap=False,
                class_weight="balanced",
            ),
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="log2",
                    max_depth=None,
                    bootstrap=False,
                ),
                n_estimators=40,
                learning_rate=0.1,
            ),
            "rusboosted_random_forests_0": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    min_samples_split=2,
                    min_samples_leaf=4,
                    max_features="log2",
                    max_depth=10,
                    bootstrap=True,
                ),
                n_estimators=30,
                learning_rate=10,
            ),
        },
    },
    "ma": {
        "scg": {
            "random_forests": RandomForestClassifier(
                n_estimators=20,
                min_samples_split=10,
                min_samples_leaf=2,
                max_features="log2",
                max_depth=None,
                bootstrap=True,
            ),
            "rusboosted_decision_trees": RUSBoostClassifier(
                estimator=DecisionTreeClassifier(
                    splitter="random",
                    min_samples_split=2,
                    min_samples_leaf=4,
                    min_impurity_decrease=0.2,
                    max_leaf_nodes=20,
                    max_features="sqrt",
                ),
                n_estimators=10,
                learning_rate=100,
            ),
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=10,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    max_depth=10,
                    bootstrap=False,
                ),
                n_estimators=20,
                learning_rate=0.1,
            ),
        },
        "mi": {
            "rusboosted_random_forests": RUSBoostClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    min_samples_split=2,
                    min_samples_leaf=4,
                    max_features="log2",
                    max_depth=None,
                    bootstrap=True,
                ),
                n_estimators=50,
                learning_rate=0.1,
            ),
        },
    },
}
