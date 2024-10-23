from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from sklearn.preprocessing   import MinMaxScaler


import pandas    as pd
import argparse
import hypertune

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--n_estimators',
        required = True,
        type     = int,
        help     = 'The number of trees in the forest')
    parser.add_argument(
        '--max_depth',
        required = True,
        type     = int,
        help     = 'The maximum depth of the tree.')
    parser.add_argument(
        '--min_samples_split',
        required = True,
        type     = int,
        help     = 'The minimum number of samples required to split an internal node')
    parser.add_argument(
        '--min_samples_leaf',
        required = True,
        type     = int,
        help     = 'The minimum number of samples required to be at a leaf node')

    args = parser.parse_args()
    return args


def create_dataset():
    '''Loads Boston Housing Data.'''

    train_data = pd.read_csv('boston_train.csv')
    train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=7)

    return train_data, validation_data


def create_model(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    '''Defines and compiles model.'''

    model = RandomForestRegressor(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
    )

    return model


def main():
    args = get_args()
    train_data, validation_data = create_dataset()
    model = create_model(args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf)
    model.fit(train_data.drop('Target', axis=1), train_data['Target'])
    predictions = model.predict(validation_data.drop('Target', axis=1))

    # DEFINE METRIC
    # As we want to minimize the mean squared error, we will negate it.
    hp_metric = mean_squared_error(predictions, validation_data['Target'])

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag = 'mean_squared_error',
        metric_value              = hp_metric)

if __name__ == "__main__":
    main()
