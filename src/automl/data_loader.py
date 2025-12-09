# lecture .data/.solution/.type
import os
import pandas as pd

def load_data_files(folder_path):
    data_file = None
    solution_file = None
    type_file = None
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.data'):
            data_file=os.path.join(folder_path, file_name)
        elif "solution" in file_name:
            solution_file=os.path.join(folder_path, file_name)
        elif file_name.endswith('.type'):
            type_file=os.path.join(folder_path, file_name)

    if data_file is not None:
        df_data=pd.read_csv(data_file, sep=' ',header=None)
        df_data.columns=[f"col{i}" for i in range(df_data.shape[1])]
    else:
        df_data=None

    if solution_file is not None:
        df_solution=pd.read_csv(solution_file, sep=' ',header=None)
        df_solution.columns=[f"col{i}" for i in range(df_solution.shape[1])]
    else:
        df_solution=None

    if type_file is not None:
        df_type=pd.read_csv(type_file,header=None)
        df_type.columns=["col1"]
    else:
        df_type=None

    return df_data, df_solution, df_type
