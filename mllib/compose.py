from sklearn.compose import make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


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
