from sklearn.compose import make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder


def make_mixed_transformer(categorical_feature_names, numerical_feature_names):
    categorical_transformer = make_pipeline(
        SimpleImputer(fill_value='missing', strategy='constant'),
        OneHotEncoder()
    )

    numerical_transformer = make_union(
        make_pipeline(
            SimpleImputer(),
            KBinsDiscretizer()
        ),
        MissingIndicator(sparse=True)
    )

    return make_pipeline(
        make_column_transformer(
            (categorical_transformer, categorical_feature_names),
            (numerical_transformer, numerical_feature_names)
        ),
        VarianceThreshold()
    )
