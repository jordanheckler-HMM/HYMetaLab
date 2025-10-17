from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def granger_two_var(df, max_lag=4):
    # df must have columns ['CCI','Rc']
    res = dict()
    X = df[["CCI", "Rc"]].to_numpy(dtype=float)
    try:
        model = VAR(X)
        fitted = model.fit(maxlags=max_lag)
        # test causality using .test_causality on fitted (requires endog names)
        # We will use the summary of coefficients via test_causality
        # statsmodels needs a DataFrame with column names
        df2 = pd.DataFrame(X, columns=["CCI", "Rc"])
        model2 = VAR(df2)
        fitted2 = model2.fit(maxlags=max_lag)
        test1 = fitted2.test_causality("CCI", ["Rc"], kind="f")
        test2 = fitted2.test_causality("Rc", ["CCI"], kind="f")

        # safe attribute access for different statsmodels versions
        def _pval(t):
            return getattr(t, "pvalue", getattr(t, "pvalues", None))

        def _stat(t):
            return getattr(
                t,
                "statistic",
                getattr(t, "test_statistic", getattr(t, "test_stat", None)),
            )

        res["CCI_to_Rc_p"] = float(_pval(test1) or 1.0)
        res["CCI_to_Rc_stat"] = float(_stat(test1) or 0.0)
        res["Rc_to_CCI_p"] = float(_pval(test2) or 1.0)
        res["Rc_to_CCI_stat"] = float(_stat(test2) or 0.0)
        res["selected_lag"] = int(fitted2.k_ar)
    except Exception:
        # retry: downsample by 2
        try:
            X2 = X[::2]
            df3 = pd.DataFrame(X2, columns=["CCI", "Rc"])
            model3 = VAR(df3)
            fitted3 = model3.fit(maxlags=max_lag)
            t1 = fitted3.test_causality("CCI", ["Rc"], kind="f")
            t2 = fitted3.test_causality("Rc", ["CCI"], kind="f")
            res["CCI_to_Rc_p"] = float(_pval(t1) or 1.0)
            res["CCI_to_Rc_stat"] = float(_stat(t1) or 0.0)
            res["Rc_to_CCI_p"] = float(_pval(t2) or 1.0)
            res["Rc_to_CCI_stat"] = float(_stat(t2) or 0.0)
            res["selected_lag"] = int(fitted3.k_ar)
        except Exception:
            # final fallback: difference once
            Xd = np.diff(X, axis=0)
            df4 = pd.DataFrame(Xd, columns=["CCI", "Rc"])
            model4 = VAR(df4)
            fitted4 = model4.fit(maxlags=max_lag)
            t1 = fitted4.test_causality("CCI", ["Rc"], kind="f")
            t2 = fitted4.test_causality("Rc", ["CCI"], kind="f")
            res["CCI_to_Rc_p"] = float(_pval(t1) or 1.0)
            res["CCI_to_Rc_stat"] = float(_stat(t1) or 0.0)
            res["Rc_to_CCI_p"] = float(_pval(t2) or 1.0)
            res["Rc_to_CCI_stat"] = float(_stat(t2) or 0.0)
            res["selected_lag"] = int(fitted4.k_ar)
    return res


def save_granger_heatmap(res, path, dpi=110):
    # res: dict with p-values
    matrix = np.zeros((2, 2))
    matrix[0, 1] = -np.log10(max(res.get("CCI_to_Rc_p", 1e-10), 1e-10))
    matrix[1, 0] = -np.log10(max(res.get("Rc_to_CCI_p", 1e-10), 1e-10))
    plt.figure(figsize=(3, 3), dpi=dpi)
    plt.imshow(matrix, cmap="magma", origin="lower")
    plt.colorbar()
    plt.xticks([0, 1], ["CCI", "Rc"])
    plt.yticks([0, 1], ["CCI", "Rc"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
