"""Unit test for preproces.py"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from src.dataloader.preprocess import (
    CustomFeatureSubtract,
    col_transform,
    label_aggregate,
    load_data,
    load_df,
    make_dataset,
)


@pytest.fixture
def mock_pd_read_csv(mocker):
    """
    Mocker that mocks pd.read_csv function in src/dataloader/preprocess.py module
    """
    mocker.patch(
        "src.dataloader.preprocess.pd.read_csv",
        side_effect=[
            pd.DataFrame({"col": [num for num in range(10)], "quality": [num for num in range(10)]}),
            pd.DataFrame({"col": [num for num in range(20)], "quality": [num for num in range(20)]}),
        ],
    )


@pytest.mark.usefixtures("mock_pd_read_csv")
class TestDataLoader:
    """Test class for DataLoader class"""

    def test_load_data_loads_two_dataframes(self):
        """Testing _load_data function actually loads two dataframes"""
        df_1, df_2 = load_data()
        assert len(df_1) > 0, "First DataFrame is empty"
        assert len(df_2) > 0, "Second DataFrame is empty"

    def test_load_df_loads_red_and_white(self):
        """Testing _load_df function loads type column with red and white"""
        df = load_df()
        type_col = list(df["type"].unique())
        assert type_col == ["red", "white"]

    def test_make_dataset_quality_is_integer(self):
        """Testing make_dataset function gives integer data type for label"""
        (_, _, y_train, y_test) = make_dataset()
        assert y_train.dtypes == int
        assert y_test.dtypes == int

    def test_make_dataset_X_does_not_have_labels(self):
        """Testing make_dataset function splits train set which shouldn't contain label"""
        (X_train, X_test, _, _) = make_dataset()
        assert "quality" not in X_train.columns
        assert "quality" not in X_test.columns

    def test_make_dataset_train_test_split(self):
        """Testing make_dataset function splits train and test set 90/10 ratio"""
        (_, _, y_train, y_test) = make_dataset()
        assert len(y_test) == np.round(0.1 * (len(y_train) + len(y_test)))


test_data = [
    (pd.DataFrame({"type": ["red", "white"]})),
    (pd.DataFrame({"type": ["red", "white"]}).astype("string")),
    (pd.DataFrame({"type": ["red", "white"]}).astype("bool")),
    (pd.DataFrame({"type": ["red", "white"]}).astype("category")),
]


class TestFeatureEngineering:
    """Test class for FeatureEngineering class"""

    def test_custom_feature_subtract_fit(self):
        """
        Testing CustomFeatureSubtract custom sklearn transformer fit method
        """
        pass

    def test_custom_feature_subtract_transform(self):
        """
        Testing CustomFeatureSubtract customer sklearn transformer transform method
        """
        custom_class = CustomFeatureSubtract("test_col_1", "test_col_2")
        _test_data = pd.DataFrame({"test_col_1": [1.0], "test_col_2": [2.0]})
        _test_expected = pd.DataFrame(
            {
                "test_col_1": [1.0],
                "test_col_2": [2.0],
                "test_col_2 minus test_col_1": [1.0],
            }
        )

        assert_frame_equal(custom_class.transform(_test_data), _test_expected)

    @pytest.mark.parametrize("input_df", test_data)
    def test_col_transform_identify_categorical_features(self, input_df):
        """
        Testing col_transform detects categorical features

        Test Cases:
        1. 'object' included in categorical features
        2. 'string' included in categorical features
        3. 'bool' included in categorical features
        4. 'category' included in categorical features
        """

        df_trans, _ = col_transform(input_df)
        assert len(df_trans) != 0

    def test_col_transform_one_hot_encoding(self):
        """
        Testing col_transform correctly gives one hot encoding

        Test Cases:
        1. column 'type' should have either 0 or 1
        """
        _test_data = pd.DataFrame({"type": ["red", "white", "red", "red"]})
        _test_expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        df_trans, _ = col_transform(_test_data)

        assert_almost_equal(df_trans, _test_expected, decimal=10)

    def test_col_transform_imputes_nan_values(self):
        """
        Testing col_transform imputes nan_values

        Test Cases:
        1. 'np.nan' should be imputed
        2. 'None' should be imputed
        """
        _test_data = pd.DataFrame(
            {"col": [np.nan, 1, 1, None], "total sulfur dioxide": [1] * 4, "free sulfur dioxide": [1] * 4}
        )

        df_trans, _ = col_transform(_test_data)
        assert np.isnan(df_trans["numeric__col"]).any() == False

    def test_col_transform_standardizes_numeric_columns(self):
        """
        Testing col_transform standardizes (mean=0, variance=1) numceric column

        Test Cases:
        1. [-1, 0, 1] retuns z=(x-u)/s where u=0, s=np.sqrt(2/3)
        """

        _test_data = pd.DataFrame(
            {"col": [-1, 0, 1], "total sulfur dioxide": [1] * 3, "free sulfur dioxide": [1] * 3}
        )
        _test_expected = pd.DataFrame(
            {
                "numeric__col": [-1 / np.sqrt(2 / 3), 0, 1 / np.sqrt(2 / 3)],
                "numeric__total sulfur dioxide": [0.0, 0.0, 0.0],
                "numeric__free sulfur dioxide": [0.0, 0.0, 0.0],
                "numeric__total sulfur dioxide minus free sulfur dioxide": [0.0, 0.0, 0.0],
            }
        )

        df_trans, _ = col_transform(_test_data)
        assert_frame_equal(df_trans, _test_expected)

    def test_label_aggregate_mapped_to_zero_one_two(self):
        """
        Testing label_aggregate changes quality integers to 0,1,2

        Test Cases:
        1. [1,2,3,4,5,6,7,8,9] gives [0,0,0,0,1,1,2,2,2]
        """
        _test_data = pd.Series(np.linspace(1, 9, num=9))
        _test_expected = pd.Series([0] * 4 + [1] * 2 + [2] * 3)

        df_label = label_aggregate(_test_data)
        assert_series_equal(df_label, _test_expected)

    def test_label_aggregate_raises_value_error_smaller_than_one_greater_than_nine(self):
        """
        Testing label_aggregate raises ValueError when qulity integer is outside [1,9]

        Test Cases:
        1. [0,10] raises ValueError
        """
        _test_data = pd.Series([0, 10])

        with pytest.raises(ValueError):
            _ = label_aggregate(_test_data)
