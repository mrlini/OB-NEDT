"""
code for the paper:
Log-loss boosting optimization with a Nash Equilibrium decision tree


Use of an AdaBoost model and a log-loss optimization mechanism to improve
the performance of an equilibrium-based decision tree. The two-step algorithm
builds equilibrium decision trees on weighted data in the first step;
during the second step, it determines the contribution of each classifier
by optimizing the overall log-loss function.
"""

import math
import numpy as np
from scipy.optimize import minimize
import typing
from sklearn.metrics import log_loss


def get_entropy(some_value):
    """Compute entropy for a given value.

    Args:
        ceva (float|list|np.ndarray): value

    Returns:
        float: entropy
    """
    if isinstance(some_value, (list, np.ndarray)):
        y = some_value
        if len(y) != 0:
            p = sum(y) / len(y)
        else:
            return 1
    else:
        p = some_value
    if (p != 0) and (p != 1):
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return 0


def get_entropy_two(y_left, y_right):
    """Get entropy for two values as in Zaki Meira, page 489.

    Args:
        y_left (float|list|np.ndarray): labels
        y_right (float|list|np.ndarray): labels

    Returns:
        float: entropy
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right
    return n_left / n * get_entropy(y_left) + n_right / n * get_entropy(y_right)


def gini(some_value):
    """Compute the gini index for a given value.

    Args:
        some_value (float|list|np.ndarray): value

    Returns:
        float: gini index
    """
    if isinstance(some_value, (list, np.ndarray)):
        y = some_value
        if len(y) != 0:
            p = sum(y) / len(y)
        else:
            return 1
    else:
        p = some_value
    return 1 - (p**2 + (1 - p) ** 2)


def gini_instances_weighted(some_value, weights):
    """Compute the weighted Gini index for a given value.

    Args:
        some_value (float|list|np.ndarray): value
        weights (_type_): weights

    Returns:
        float: weighted Gini index
    """
    n_classes = np.unique(some_value)
    p = np.zeros(len(n_classes))
    if isinstance(some_value, (list, np.ndarray)):
        y = some_value
        if len(y) != 0:
            for cl in range(len(n_classes)):
                p[cl] = np.sum(weights[some_value == n_classes[cl]]) / np.sum(weights)

        else:
            return 1
    else:
        p = some_value
    return 1 - np.sum(p**2)


def weighted_gini(y_left, y_right, weights_left, weights_right):
    """zaki meira page 489

    Args:
        y_left (float|list|np.ndarray): labels
        y_right (float|list|np.ndarray): labels
        weights_left (_type_): weights
        weights_right (_type_): weights

    Returns:
        float: weighted gini
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right
    return n_left / n * gini_instances_weighted(
        y_left, weights_left
    ) + n_right / n * gini_instances_weighted(y_right, weights_right)


class Node:
    """Node for OB-NEDT"""

    def __init__(
        self,
        criterion,
        split_criterion,
        num_samples,
        num_samples_per_class,
        predicted_class,
        predicted_class_prob,
        k_quantile_zero=0.2,
    ):
        self.criterion = criterion
        self.split_criterion = split_criterion
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.predicted_class_prob = predicted_class_prob
        self.k_quantile_zero = k_quantile_zero
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.beta = 0
        self.attribute = -1

    def gain_entropy_for_opt(self, cut, prod, y):
        """entropy for optimization"""
        y_left = y[prod <= cut]
        y_right = y[prod > cut]
        return -(get_entropy(y) - get_entropy_two(y_left, y_right))

    def gini_for_opt(self, cut, prod, y, weights):
        """gini for optimization"""
        y_left = y[prod <= cut]
        weights_left = weights[prod <= cut]
        y_right = y[prod > cut]
        weights_right = weights[prod > cut]
        return weighted_gini(y_left, y_right, weights_left, weights_right)

    def split_atr_stg_dr_cu_entr_gini_si_opt(self, X, y, beta, feature, weights):
        """split with entropie/gini and opt

        Args:
            X (_type_): data instances to split
            y (_type_): labels
            beta (_type_): beta parameter
            feature (_type_): _description_

        Returns:
            _type_: _description_
        """
        prod = X[:, feature] * beta[1] + beta[0]
        indexes = np.argsort(prod)
        ordered_prod = prod[indexes].copy()
        y_ordered = y[indexes].copy()

        # find cut such that Gain is maximized
        if self.split_criterion == "entropy":
            functie = self.gain_entropy_for_opt
        elif self.split_criterion == "gini":
            functie = self.gini_for_opt

        prod0 = np.quantile(prod[y == 0], self.k_quantile_zero)
        prod1 = np.quantile(prod[y == 1], 1 - self.k_quantile_zero)
        cut = (prod0 + prod1) / 2

        y_left = y_ordered[ordered_prod <= cut]
        weights_left = weights[ordered_prod <= cut]
        y_right = y_ordered[ordered_prod > cut]
        weights_right = weights[ordered_prod > cut]
        return y_left, y_right, cut, weights_left, weights_right

    def entropy_gain(self, X, y, beta, attribute, weights):
        """Compute y_left, y_right and gain, use entropy"""
        (
            y_left,
            y_right,
            cut,
            weights_left,
            weights_right,
        ) = self.split_atr_stg_dr_cu_entr_gini_si_opt(X, y, beta, attribute, weights)

        return get_entropy(y) - get_entropy_two(y_left, y_right), cut

    def gini_gain(self, X, y, beta, attribute, weights):
        """Compute y_left, y_right and gain, use gini"""
        (
            y_left,
            y_right,
            cut,
            weights_left,
            weights_right,
        ) = self.split_atr_stg_dr_cu_entr_gini_si_opt(X, y, beta, attribute, weights)

        return -weighted_gini(y_left, y_right, weights_left, weights_right), cut

    def split_attribute(self, X, y, weights):
        """Split

        Args:
            X (np.ndarray): data instances
            y (np.ndarray): true labels

        Returns:
            _type_: _description_
        """
        n_attributes = X.shape[1]
        Beta = np.zeros((n_attributes, 2))
        Gain = np.zeros(n_attributes)
        cuts = np.zeros(n_attributes)
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        w0 = np.sum(weights[y == 0]) / np.sum(weights)
        w1 = np.sum(weights[y == 1]) / np.sum(weights)

        beta_new = np.zeros(2)

        if self.split_criterion == "entropy":
            function_used = self.entropy_gain
        elif self.split_criterion == "gini":
            function_used = self.gini_gain

        for attribute_i in range(n_attributes):
            Xatribut = X[:, attribute_i].copy()
            beta_new[0] = 1 / 8 * (n0 / w0 - n1 / w1)
            beta_new[1] = 1 / 8 * np.sum(Xatribut * weights)
            Beta[attribute_i] = beta_new.copy()

            Gain[attribute_i], cuts[attribute_i] = function_used(
                X, y, Beta[attribute_i], attribute_i, weights
            )
        self.attribute = np.argmax(Gain)
        self.beta = Beta[self.attribute].copy()
        self.threshold = cuts[self.attribute]

        return self.beta, self.attribute, self.threshold

    def split_attribute_no_game(self, X, y, weights):
        """Split without game"""
        n_attributes = X.shape[1]
        Beta = np.zeros((n_attributes, 2))
        Gain = np.zeros(n_attributes)
        cuts = np.zeros(n_attributes)
        beta_new = np.zeros(2)

        if self.split_criterion == "entropy":
            function_used = self.entropy_gain
        elif self.split_criterion == "gini":
            function_used = self.gini_gain

        for attribute_i in range(n_attributes):
            beta_new[0] = 0
            beta_new[1] = 1
            Beta[attribute_i] = beta_new.copy()

            Gain[attribute_i], cuts[attribute_i] = function_used(
                X, y, Beta[attribute_i], attribute_i, weights
            )

        self.attribute = np.argmax(Gain)
        self.beta = Beta[self.attribute].copy()
        self.threshold = cuts[self.attribute]
        return self.beta, self.attribute, self.threshold


class OBNEDT:
    """OB-NEDT Nash equilibrium based decision tree classifier"""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = None,
        game_based: int = 1,
        k_quantile_zero: float = 0.2,
    ):
        """constructor

        Args:
            criterion (str, optional): split criterion. Defaults to 'gini'.
            max_depth (int, optional): max depth for decision tree. Defaults to None.
            game_based (int, optional): use game. Defaults to 0.
            k_quantile_zero (float, optional): quantile. Defaults to 0.2.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.game_based = game_based
        self.nr_nodes = 0
        self.k_quantile_zero = k_quantile_zero
        self.n_classes_ = 0
        self.n_features_ = 0
        self.tree_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, weights):
        """fit classifier

        Args:
            X (np.ndarray): train instances
            y (np.ndarray): train labels
            weights (np.ndarray): weights for each instance
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, weights, 0)

    def predict(self, X: np.ndarray):
        """Make predictions for X.

        Args:
            X (np.ndarray): test instances

        Returns:
            np.ndarray, np.ndarray: predicted class, predicted class
              probability for each test instance
        """
        pred_class = list()
        pred_prob = list()
        for data_instance in X:
            a, b = self._predict_instance(data_instance)

            pred_class.append(a)
            pred_prob.append(b)
        return pred_class, pred_prob

    def _predict_instance(self, inputs: np.ndarray):
        """Predict the class for one instance.

        Args:
            inputs (np.ndarray): data instance

        Returns:
            tuple: predicted class, predicted class probability
        """
        node = self.tree_
        while node.left:
            if inputs[node.attribute] * node.beta[1] + node.beta[0] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class, node.predicted_class_prob

    def _grow_tree(self, X, y, weights, depth=0):
        """Build a decision tree by recursively finding the best split.

        Args:
            X (np.ndarray): data instances
            y (np.ndarray): true labels for X
            depth (int, optional): max depth for the tree. Defaults to 0.
        """
        # Population for each class in current node.
        # The predicted class is the one with most instances in population.
        num_samples_per_class = [
            np.sum(weights[y == i]) for i in range(self.n_classes_)
        ]
        predicted_class = np.argmax(num_samples_per_class)
        # probability for predicted class 1
        predicted_class_prob = num_samples_per_class[1] / np.sum(weights)
        atr_node = 0

        node = Node(
            criterion=atr_node,
            split_criterion=self.criterion,
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            predicted_class_prob=predicted_class_prob,
            k_quantile_zero=self.k_quantile_zero,
        )

        # Split recursively until maximum depth is reached.
        if (depth < self.max_depth) & (X.shape[0] >= 1) & np.all(num_samples_per_class):
            if self.game_based == 1:
                node.split_attribute(X, y, weights)
            else:
                node.split_attribute_no_game(X, y, weights)
            produs = X[:, node.attribute] * node.beta[1] + node.beta[0]

            X_left, y_left, weights_left = (
                X[produs <= node.threshold],
                y[produs <= node.threshold],
                weights[produs <= node.threshold],
            )
            X_right, y_right, weights_right = (
                X[produs > node.threshold],
                y[produs > node.threshold],
                weights[produs > node.threshold],
            )

            if len(y_left) == 0 or len(y_right) == 0:
                print(" ")
            else:
                if len(y_left) > 0:
                    node.left = self._grow_tree(X_left, y_left, weights_left, depth + 1)
                if len(y_right) > 0:
                    node.right = self._grow_tree(
                        X_right, y_right, weights_right, depth + 1
                    )

        return node


def sub_sample_probability(
    X: np.ndarray, y: np.ndarray, p: np.ndarray, sample_size: float = 0.7
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample the data.

    Args:
        X (np.ndarray): data instances
        y (np.ndarray): true labels
        p (np.ndarray): probability to choose an instances from the next tree
        sample_size (float, optional): sample size, in (0, 1.0]. Defaults to 0.7.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: data instances, true labels and indexes
    """
    n = X.shape[0]
    n_subsample = int(n * sample_size)
    idx = np.random.choice(n, size=n_subsample, p=(1 - p) / np.sum(1 - p))
    return X[idx].copy(), y[idx].copy(), idx


def update_weights(
    y_train_subsample: np.ndarray,
    weights: np.ndarray,
    nedt_pred_train: typing.List[int],
    indexes: np.ndarray,
) -> typing.Tuple[np.ndarray, float]:
    """
    Update weights. After AdaBoost, as in Statistical Learning.

    Args:
        y_train_subsample (np.ndarray): labels for the subsample
        weights (np.ndarray): weights for each instance
        nedt_pred_train (typing.List[int]): predicted labels for the subsample
        indexes (np.ndarray): indexes of the subsample

    Returns:
        typing.Tuple[np.ndarray, float]: updated weights and alpha
    """
    indexes = np.array(indexes)

    err_m = np.sum(weights[indexes] * (y_train_subsample != nedt_pred_train)) / np.sum(
        weights[indexes]
    )
    if (err_m != 0) & (err_m != 1):
        alfa_m = np.log((1 - err_m) / err_m)
        new_weights = weights.copy()
        new_weights[indexes[y_train_subsample != nedt_pred_train]] *= np.exp(alfa_m)
    else:
        alfa_m = 0
        new_weights = np.ones_like(weights)

    return (new_weights) / np.max(new_weights), alfa_m


def log_loss_weights(weights_trees, y, probs):
    """
    Compute log loss for each tree and then average them with weights

    Args:
        weights_trees: weights for each tree in log loss, to minimize
        y: labels
        probs: matrix, trees on rows, columns store probabilities for each instance

    Returns:
        float: log loss to maximize
    """
    p = np.average(probs, weights=weights_trees, axis=0)
    return log_loss(y, p)


def run_ob_nedt(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    n_estimators_rf: int = 50,
    n_subset_var_rf: float = 0.7,
    sample_size_tree: float = 0.7,
    split_criterion_tree: str = "gini",
    max_depth_tree: int = 5,
    game_tree: bool = True,
    k_quantile_zero_tree: float = 0.2,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Optimized Boosting with Nash Equilibrium Decision Trees

    For simplicity, fit() and predict() are defined in this function.

    Return the predicted class and the predicted probability for each
    instance in the test set.

    Args:
        X_train (np.ndarray): train instances
        X_test (np.ndarray): test instances
        y_train (np.ndarray): true labels for X_train
        n_estimators_rf (int, optional): number of estimators for RF NEDT. Defaults to 50.
        n_subset_var_rf (float, optional): number of subsets. Defaults to 0.7.
        sample_size_tree (float, optional): sample size chosen for one tree of the RF. Defaults to 0.7.
        split_criterion_tree (str, optional): split criterion for the tree. Defaults to "gini".
        max_depth_tree (int, optional): tree max depth. Defaults to 5.
        game_tree (bool, optional): use game in tree. Defaults to True.
        k_quantile_zero_tree (float, optional): k quantile zero DT. Defaults to 0.2.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: predicted class and predicted
        probability for each instance in the test set
    """
    n_pred = X_test.shape[0]
    n_atr = X_train.shape[1]
    n_atr_subsample = int(n_atr * n_subset_var_rf)
    # used to store predictions for each tree in the forest
    rf_nedt_pred_temp = np.zeros((n_estimators_rf, n_pred), dtype=np.int64)
    rf_nedt_pred_prob_temp = np.zeros((n_estimators_rf, n_pred), dtype=np.float64)
    # used for the final prediction
    rf_nedt_pred = np.zeros(n_pred, dtype=np.int64)
    rf_nedt_pred_prob = np.zeros(n_pred)

    rf_nedt_pred_train = np.zeros((n_estimators_rf, X_train.shape[0]), dtype=np.int64)
    rf_nedt_pred_prob_train = np.zeros((n_estimators_rf, X_train.shape[0]))

    n_samples_training = X_train.shape[0]
    probabilities = np.ones(n_samples_training) / n_samples_training
    weights = np.ones(n_samples_training) / n_samples_training

    alfa_m = np.zeros(n_estimators_rf)

    for tree_i in range(n_estimators_rf):
        X_train_subsample, y_train_subsample, indexes = sub_sample_probability(
            X_train, y_train, probabilities, sample_size_tree
        )
        X_test_copac = X_test.copy()

        if (n_atr > 2) and (n_atr_subsample < n_atr) and (n_atr_subsample > 0):
            cols_to_delete = np.random.choice(
                n_atr, size=(n_atr - n_atr_subsample), replace=False
            )
            X_train_subsample = np.delete(X_train_subsample, cols_to_delete, axis=1)
            X_test_copac = np.delete(X_test_copac, cols_to_delete, axis=1)

        clf_nedt = OBNEDT(
            criterion=split_criterion_tree,
            max_depth=max_depth_tree,
            game_based=game_tree,
            k_quantile_zero=k_quantile_zero_tree,
        )

        clf_nedt.fit(X_train_subsample, y_train_subsample, weights[indexes])
        nedt_pred_train, nedt_pred_prob_train = clf_nedt.predict(X_train)

        rf_nedt_pred_train[tree_i, :], rf_nedt_pred_prob_train[tree_i, :] = (
            nedt_pred_train.copy(),
            nedt_pred_prob_train.copy(),
        )

        weights, alfa_tree = update_weights(
            y_train,
            weights,
            nedt_pred_train,
            np.arange(len(y_train)),
        )

        alfa_m[tree_i] = alfa_tree
        (
            rf_nedt_pred_temp[tree_i, :],
            rf_nedt_pred_prob_temp[tree_i, :],
        ) = clf_nedt.predict(X_test_copac)

    # bagging and predict
    for i in range(n_pred):
        pred_nedt = rf_nedt_pred_temp[:, i].tolist()
        rf_nedt_pred[i] = max(set(pred_nedt), key=pred_nedt.count)
        rf_nedt_pred_prob[i] = np.mean(rf_nedt_pred_prob_temp[:, i])

    # optimize
    res = minimize(
        log_loss_weights,
        alfa_m,
        args=(y_train, rf_nedt_pred_prob_train),
        method="Powell",
    )
    res_x = res.x.copy()

    alfa_m_w = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    rf_nedt_pred_prob = np.average(rf_nedt_pred_prob_temp, weights=alfa_m_w, axis=0)
    rf_nedt_pred = np.rint(rf_nedt_pred_prob)

    return rf_nedt_pred, rf_nedt_pred_prob
