import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def make_ds(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        lines: list[str] = f.readlines()

    variables: int = int(lines[0].strip())
    lines.pop(0)
    instances: int = int(lines[0].strip())
    lines.pop(0)

    X: torch.Tensor = torch.zeros(variables, instances)
    y: torch.Tensor = torch.zeros(instances)

    for index_line in range(len(lines)):
        list_instance: list[str] = lines[index_line].split()
        for index_el in range(len(list_instance) - 1):
            X[index_el, index_line] = float(list_instance[index_el])
        y[index_line] = float(list_instance[-1])

    return X.T, y


def make_train_test_ds(name_ds: str, number_test: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path_train: str = "datasets/" + name_ds + "/train" + str(number_test)
    path_test: str = "datasets/" + name_ds + "/test" + str(number_test)

    X_train, y_train = make_ds(path_train)
    X_test, y_test = make_ds(path_test)

    return X_train, y_train, X_test, y_test

def read_csv_data(folder_path: str, dataset_name: str, idx: int, scale_strategy: str = 'no') -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not folder_path.endswith('/'):
        raise AttributeError(f'Provided folder path does not end with /.')
    if scale_strategy not in ('no', 'standard', 'robust', 'minmax'):
        raise AttributeError(f'{scale_strategy} is an invalid scale strategy. Allowed ones: no, standard, robust, minmax.')

    d: pd.DataFrame = pd.read_csv(folder_path+dataset_name+'/'+'train'+str(idx)+'.csv')
    y: np.ndarray = d['target'].to_numpy()
    d.drop('target', axis=1, inplace=True)
    X: np.ndarray = d.to_numpy()
    
    if scale_strategy == 'standard':
        data_scaler: StandardScaler = StandardScaler()
    elif scale_strategy == 'robust':
        data_scaler: RobustScaler = RobustScaler()
    elif scale_strategy == 'minmax':
        data_scaler: MinMaxScaler = MinMaxScaler()    

    if scale_strategy != 'no':
        data_scaler = data_scaler.fit(X)
        X = data_scaler.transform(X)

    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {'train': (torch.from_numpy(X), torch.from_numpy(y))}
    
    d = pd.read_csv(folder_path+dataset_name+'/'+'test'+str(idx)+'.csv')
    y = d['target'].to_numpy()
    d.drop('target', axis=1, inplace=True)
    X = d.to_numpy()

    if scale_strategy != 'no':
        X = data_scaler.transform(X)

    result['test'] = (torch.from_numpy(X), torch.from_numpy(y))
    
    return result

