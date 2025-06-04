import numpy as np 
import pandas as pd

data = pd.read_csv("Agente_saude/tabela_sintomas.csv", header=None, skiprows=1)
data.drop(data.columns[[0]], axis=1, inplace=True)

data.replace({
    'comum': 1,
    'as vezes': 2,
    'raro': 3
}, inplace=True)
data.infer_objects(copy=False)


class Node:
    count=0
    def __init__(self, left, right, data=None, feature_idx=None, 
                 threshold=None, feature_name=None, 
                 criterion_value=None, _result=None, 
                 class_name=None):
        self.id=Node.count
        Node.count += 1
    
        self.left: Node = left
        self.right: Node = right

        self.data = data
        self.feature_idx: int = feature_idx
        self.feature_name: str = feature_name
        self.threshold: float = threshold
        self.criterion_value: float = criterion_value
        self.n_sample: int = len(data) if data is not None else 0
        self._result = _result
        self.class_name: str = class_name

    def predict(self, x):
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        
        elif x[self.feature_idx] >= self.threshold:
            return self.right.predict(x)


class LeafNode(Node):
    def __init__(self, data, criterion_value, _result, class_name):
        super().__init__(None, None, data=data, 
                         criterion_value=criterion_value, 
                         _result=_result, class_name=class_name)

    def predict(self, X=None):
        return self._result






class decision_tree():
    def __init__(self, max_depth=5, min_samples_to_split=4, min_samples_leaf=2,
                 feature_names=[], class_names=[]):
        
        self.root: Node = Node(None, None)#: Node é uma anotação de tipo (type hint) Isso indica que o atributo self.root será do tipo Node — ou seja, uma instância da classe Node.
                                          #Essa anotação não executa nada, apenas documenta o tipo esperado
    
                # hyperparameters
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.min_samples_leaf = min_samples_leaf

        self.feature_names = feature_names
        self.class_names = class_names
        
    def _split_data(self, feature_data, labels, thresh: float):
        # splitting left
        left_indices = feature_data < thresh
        left_feature_data = feature_data[left_indices]
        left_labels = labels[left_indices]

    # splitting right
        right_indices  = feature_data >= thresh
        right_feature_data = feature_data[right_indices]
        right_labels = labels[right_indices]

        return (left_feature_data, left_labels), (right_feature_data, right_labels)

    def _best_split(self, feature_data, labels, thresholds):
        min_split_cost = np.inf
        min_cost_thresh = None
    # for each threshold
        for thresh in thresholds:
            left_data, right_data = self._split_data(feature_data, labels, thresh)

            left_labels = left_data[1]
            right_labels = right_data[1]

            cost_function = impurity_function
           
        # calculate the partitioning criterion for the feature with the specific threshold
            cost_value = cost_function(left_labels, right_labels)
        
            if cost_value < min_split_cost:
                min_split_cost = cost_value
                min_cost_thresh = thresh
    
        return min_cost_thresh, min_split_cost

    def _best_feature(self, data, feature_idxs):

        min_feature_cost = np.inf
        min_feature_threshold = None
        selected_feature = None
    # for each feature in a list of the features index
        for feature_idx in feature_idxs:

        # get the feature data
            feature_data = data[:, feature_idx]
            labels = data[:, -1]
        
            unique_values = np.sort(np.unique(feature_data))
        
        # generate the thresholds
            thresholds = mean_adjacent(unique_values, window_size=2)
            min_thresh, min_split_cost = self._best_split(feature_data, labels, thresholds)

            if min_split_cost < min_feature_cost:
                min_feature_cost = min_split_cost
                min_feature_threshold = min_thresh
                selected_feature = feature_idx


    def _grow(self, data, depth=1):
        compute_criterion_value = gini_impurity
        get_result = get_majority_class

        y = data[:, -1]
    # Calculate the criterion value of the data
        criterion_value = compute_criterion_value(y)
        result = get_result(y)

        class_name = self.class_names[result]

    # Stopping criteria
        if self.max_depth and depth >= self.max_depth:
            return LeafNode(data, criterion_value=criterion_value, 
                        _result = result, class_name=class_name)
    
        if len(data) < self.min_samples_to_split:
            return LeafNode(data, criterion_value=criterion_value, 
                        _result=result, class_name=class_name)
    
        if criterion_value < EPSILON:
            return LeafNode(data, criterion_value=criterion_value, 
                        _result=result,class_name=class_name)
    
        feature_idxs = np.arange(data.shape[-1] - 1)

    # Splitting
        selected_feature, min_feature_threshold, _ = self._best_feature(data, feature_idxs)
    
    # Split data based on best split
        left_data = data[data[:, selected_feature] < min_feature_threshold]
        right_data = data[data[:, selected_feature] >= min_feature_threshold]

    # Stopping criterion based on the length of the child data
        if (len(left_data) < self.min_samples_leaf 
        or len(right_data) < self.min_samples_leaf):
            return LeafNode(data, criterion_value=criterion_value, 
                        _result=result, class_name=class_name)

    # Create child nodes
        left_node = self._grow(left_data, depth=depth+1)
        right_node = self._grow(right_data, depth=depth+1)

        return Node(left_node, 
                right_node,
                data, 
                selected_feature, 
                min_feature_threshold, 
                feature_name=self.feature_names[selected_feature] if any(self.feature_names) else "NA",
                criterion_value = criterion_value,
                _result=result, class_name=class_name)


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._grow(np.hstack((X, y)))

    def predict(self, X):
        return self.root.predict(X)

