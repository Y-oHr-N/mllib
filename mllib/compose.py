from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from .utils import get_categorical_columns
from .utils import get_numerical_columns


def make_categorical_transformer(X_type=None, y_type=None, encode='onehot'):
    imputer = SimpleImputer(fill_value='missing', strategy='constant')

    if encode == 'onehot':
        encoder = OneHotEncoder()
    elif encode == 'ordinal':
        encoder = OrdinalEncoder()
    else:
        raise ValueError(f'unknown encode: {encode}')

    return make_pipeline(imputer, encoder)


def make_numerical_transformer(X_type=None, y_type=None, scale='standard'):
    imputer = SimpleImputer()

    if scale == 'maxabs':
        scaler = MaxAbsScaler()
    elif scale == 'minmax':
        scaler = MinMaxScaler()
    elif scale == 'robust':
        scaler = RobustScaler()
    elif scale == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f'unknown scale: {scale}')

    return make_pipeline(imputer, scaler)


def make_mixed_transformer(
    X_type=None,
    y_type=None,
    encode='onehot',
    scale='standard'
):
    categorical_transformer = make_categorical_transformer(
        X_type,
        y_type,
        encode
    )
    numerical_transformer = make_numerical_transformer(X_type, y_type, scale)

    return make_column_transformer(
        (categorical_transformer, get_categorical_columns),
        (numerical_transformer, get_numerical_columns)
    )


def make_preprocessor(
    X_type='numerical',
    y_type=None,
    encode='onehot',
    scale='standard'
):
    if X_type == 'categorical':
        return make_categorical_transformer(X_type, y_type, encode)
    elif X_type == 'numerical':
        return make_numerical_transformer(X_type, y_type, scale)
    elif X_type == 'mixed':
        return make_mixed_transformer(X_type, y_type, encode, scale)
    else:
        raise ValueError(f'unknown data type: {X_type}')


def make_classifer(X_type='numerical', y_type='multiclass'):
    classifier = SGDClassifier(max_iter=1000, tol=1e-03)

    if y_type in ['binary', 'multiclass']:
        return classifier
    elif y_type == 'multiclass-output':
        return MultiOutputClassifier(classifier)
    else:
        raise ValueError(f'unknown target type: {y_type}')


def make_regressor(X_type='numerical', y_type='continuous'):
    regressor = SGDRegressor(max_iter=1000, tol=1e-03)

    if y_type == 'continuous':
        return regressor
    elif y_type == 'continuous-output':
        return MultiOutputRegressor(regressor)
    else:
        raise ValueError(f'unknown target type: {y_type}')


def make_model(X_type, y_type, encode='onehot', scale='standard'):
    """Make a benchmark model.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.utils.multiclass import type_of_target
    >>> from mllib.utils import type_of_data
    >>> X, y = load_iris(return_X_y=True)
    >>> X_type = type_of_data(X)
    >>> y_type = type_of_target(y)
    >>> model = make_model(X_type, y_type)
    """

    preprocessor = make_preprocessor(X_type, y_type, encode, scale)

    if y_type in ['binary', 'multiclass', 'multiclass-output']:
        model = make_classifer(X_type, y_type)
    elif y_type in ['continuous', 'continuous-output']:
        model = make_regressor(X_type, y_type)
    else:
        raise ValueError(f'unknown target type: {y_type}')

    return make_pipeline(preprocessor, model)
