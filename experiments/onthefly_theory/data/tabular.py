from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, fetch_california_housing, fetch_covtype
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

def load_openml_safe(name: Optional[str]=None, data_id: Optional[int]=None, version=None, as_frame=True):
    try:
        if name is not None:
            if version is None:
                try:
                    ds = fetch_openml(name=name, version="active", as_frame=as_frame)
                except TypeError:
                    ds = fetch_openml(name=name, as_frame=as_frame)
            else:
                ds = fetch_openml(name=name, version=version, as_frame=as_frame)
        else:
            if version is None:
                ds = fetch_openml(data_id=data_id, as_frame=as_frame)
            else:
                try:
                    ds = fetch_openml(data_id=data_id, version=version, as_frame=as_frame)
                except TypeError:
                    ds = fetch_openml(data_id=data_id, as_frame=as_frame)
        X = ds.data.copy(); y = ds.target.copy()
        return X, y
    except Exception as e:
        print(f"[load_openml_safe] Skipping {name or data_id}: {e}")
        return None, None

def prep_regression_frame(X: pd.DataFrame, y: pd.Series):
    X = X.copy()
    y = pd.to_numeric(pd.Series(y).astype(float), errors="coerce")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True) if "sparse_output" in OneHotEncoder().get_params() \
          else OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None
    )
    return pre

def prep_classification_frame(X: pd.DataFrame, y: pd.Series):
    X = X.copy(); y = pd.Series(y)
    if not pd.api.types.is_integer_dtype(y):
        y = y.astype("category").cat.codes
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True) if "sparse_output" in OneHotEncoder().get_params() \
          else OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None
    )
    return pre, y.astype(int).values

def build_tabular_regression(hp) -> List[Tuple[str, pd.DataFrame, pd.Series]]:
    out = []
    if hp.use_tr_california:
        cal = fetch_california_housing(as_frame=True)
        Xc, yc = cal.frame.drop(columns=["MedHouseVal"]), cal.frame["MedHouseVal"]
        out.append(("CaliforniaHousing", Xc, yc))
    if hp.use_tr_airfoil:
        X, y = load_openml_safe(name="airfoil_self_noise")
        if X is not None: out.append(("AirfoilSelfNoise", X, y))
    if hp.use_tr_concrete:
        X, y = load_openml_safe(name="Concrete Compressive Strength")
        if X is None:
            X, y = load_openml_safe(name="concrete", version=6)
        if X is not None: out.append(("ConcreteStrength", X, y))
    if hp.use_tr_energy:
        X, y = load_openml_safe(name="energy-efficiency")
        if X is not None:
            if "Y1" in X.columns:
                y1 = X["Y1"]; X1 = X.drop(columns=["Y1","Y2"], errors="ignore")
                out.append(("EnergyEfficiency_Y1", X1, y1))
            else:
                out.append(("EnergyEfficiency", X, y))
    if hp.use_tr_yacht:
        X, y = load_openml_safe(name="yacht_hydrodynamics")
        if X is not None: out.append(("YachtHydrodynamics", X, y))
    if hp.use_tr_ccpp:
        X, y = load_openml_safe(name="Combined Cycle Power Plant")
        if X is None:
            X, y = load_openml_safe(name="CCPP")
        if X is not None: out.append(("CCPP", X, y))
    return out

def build_tabular_classification(hp):
    out = []
    if hp.use_tc_adult:
        X, y = load_openml_safe(name="adult")
        if X is not None: out.append(("Adult", X, y))
    if hp.use_tc_covtype:
        cov = fetch_covtype(as_frame=True)
        Xc, yc = cov.frame.drop(columns=["Cover_Type"]), cov.frame["Cover_Type"]
        out.append(("Covertype", Xc, yc))
    return out
