"""
Created on Tue Jul 16 11:07:51 2019

@author: ctlab
@link: https://github.com/ctlab/ITMO_FS/blob/master/ITMO_FS/utils/information_theory.py
"""

import numpy as np

from operator import itemgetter
from itertools import groupby
from math import log, fsum
from collections import Counter

def entropy(x):
    """Calculate the entropy (H(X)) of an array.

    Parameters
    ----------
    x : array-like, shape (n,)
        The array.

    Returns
    -------
    float : H(X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import entropy
    >>> entropy([1,1,1])
    0.0
    >>> entropy([1,2,3,4,5])
    1.6094379124341003
    >>> entropy([5,4,1,2,3])
    1.6094379124341003
    >>> entropy([1,2,1,2,1,2,1,2,1,2])
    0.6931471805599456
    >>> entropy([1,1,1,1,1,2])
    0.4505612088663047
    """
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)


def conditional_entropy(x_j, y):
    """Calculate the conditional entropy (H(Y|X)) between two arrays.

    Parameters
    ----------
    x_j : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    float : H(Y|X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import conditional_entropy
    >>> conditional_entropy([1,2,1,3,4], [1,2,3,4,5])
    0.2772588722239781
    >>> conditional_entropy([1], [2])
    0.0
    >>> conditional_entropy([1,2,1,3,2,4], [3,3,3,3,3,3])
    0.0
    >>> conditional_entropy([1,2,3,1,3,2,3,4,1], [1,2,1,3,1,4,4,1,5])
    0.7324081924454064
    """
    buf = [[e[1] for e in g] for _, g in 
           groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x_j)


def su_measure(x, y):
    """SU is a correlation measure between the features and the class
    calculated via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y)). Bigger values
    mean more important features. This measure works best with discrete
    features due to being based on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import su_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> su_measure(x, y)
    array([0.28694182, 0.13715115, 0.79187567, 0.47435099, 0.67126949])
    """
    
    def __SU(feature):
        entropy_x = entropy(feature)
        return (2 * (entropy_x - conditional_entropy(y, feature))
                  / (entropy_x + entropy_y))

    entropy_y = entropy(y)
    return np.apply_along_axis(__SU, 0, x)