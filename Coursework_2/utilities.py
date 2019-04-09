import numpy as np

def load_data(train_set_path='data/wine_train.csv', 
              train_labels_path='data/wine_train_labels.csv', 
              test_set_path='data/wine_test.csv',
              test_labels_path='data/wine_test_labels.csv'):
    """
    Loads the wine dataset. If no arguments are passed it will try to load the data
    from the working directory with the default file names
    
    Args:
        train_set_path : path to the train set .csv file
        train_labels_path : path to the train labels .csv file
        test_set_path : path to the test set .csv file
        test_labels_path : path to the testlabels .csv file
    Returns:
        (train_set, train_labels, test_set, test_labels), numpy arrays containing the
        training and testing sets, along with the respective class labels
    """
    
    train_set = np.loadtxt(train_set_path, delimiter=',')
    train_labels = np.loadtxt(train_labels_path, delimiter=',', dtype=np.int)
    test_set = np.loadtxt(test_set_path, delimiter=',')
    test_labels = np.loadtxt(test_labels_path, delimiter=',', dtype=np.int)
    
    return train_set, train_labels, test_set, test_labels


def print_predictions(predictions):
    """
    Prints the classifier predictions to the standard output in the format expected
    by the auto-marker.
    
    Args: 
        predictions: can be either a list or a NumPy array. 
        If your predictions are an np.array, then the array must be either 1D or 
        have shape (n, 1) or (1, n),
        If your predictions are a list, then it must be a 1D list
    """
    _print_for_automaker(predictions, 'predictions')
    

def print_features(features):
    """
    Prints the selected features to the standard output in the format expected
    by the auto-marker.
    
    Args: 
        features: can be either a list or a NumPy array. 
        If your features are an np.array, then the array must be either 1D or 
        have shape (n, 1) or (1, n),
        If your features are a list, then it must be a 1D list
    """
    _print_for_automaker(features, 'features')


def _print_for_automaker(D, what):    
    """
    Internal function for printing things for the auto-marker. 
    You should not use this function. 
    Use either `print_predictions` or `print_features`
    """
    p = None
    t = type(D)
    
    if t is np.ndarray:
        assert D.ndim == 1 or (D.ndim == 2 and min(D.shape) == 1), \
        'If your {} are an np.array, then the array must be either 1D or have shape (n, 1) or (1, n). Your shape is {}'.format(what, D.shape)
        p = D.reshape(max(D.shape)).tolist()
        assert len(p) > 0, 'Empty {}!'.format(what)
    elif t is list:
        assert len(D) > 0, 'Empty {}!'.format(what)
        assert type(D[0]) is not list, 'If your {} are a list, then it must be a 1D list'.format(what)
        p = D
    else:
        raise Exception('{} should be passed as numpy array or list. Your predictions were of type {}'.format(what, t))
        
    print(p)
