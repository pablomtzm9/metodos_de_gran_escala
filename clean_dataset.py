# Funciones

import numpy as np

def clean_dataset(df):
    """Limpieza de un dataframe de: na, nan, inf

    Parámetros
    ----------
    df : dataframe
        archivo que se quiere limpiar

    Returns
    -------
    dataframe
        archivo sin los renglones que tuvieron na, nan o inf
    """
  
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)