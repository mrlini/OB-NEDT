from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from ob_nedt import run_ob_nedt


def run_main():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        random_state=42,
        class_sep=0.1,
        weights=[0.5],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=123
    )

    nedt_pred, nedt_pred_prob = run_ob_nedt(
        X_train,
        X_test,
        y_train,
        n_estimators_rf=10,
        n_subset_var_rf=0.7,
        sample_size_tree=0.7,
        split_criterion_tree="gini",
        max_depth_tree=5,
        game_tree=True,
        k_quantile_zero_tree=0.2,
    )
    rez_auc = roc_auc_score(y_test, nedt_pred_prob)
    rez_acc = accuracy_score(y_test, nedt_pred)
    rez_f1 = f1_score(y_test, nedt_pred)
    rez_log_loss = log_loss(y_test, nedt_pred_prob)
    print(f"results:\n- roc_auc_score:\t {rez_auc}")
    print(f"- accuracy_score:\t {rez_acc}")
    print(f"- f1_score:\t\t {rez_f1}")
    print(f"- log_loss:\t\t {rez_log_loss}")


if __name__ == "__main__":
    run_main()
