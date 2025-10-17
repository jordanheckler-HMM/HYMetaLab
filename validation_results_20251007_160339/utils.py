import numpy as np
import pandas as pd
import statsmodels.api as sm


def log_safe(s, name):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return np.log(s)


def fit_ols(df, y_col, x_cols):
    # Defensive: ensure df has rows and x columns
    if df is None or len(df) == 0:
        summ = {"r2": None, "r2_adj": None, "n": 0}
        empty = pd.DataFrame(
            columns=["term", "coef", "stderr", "ci_low", "ci_high", "pval"]
        )
        return None, summ, empty
    try:
        X = sm.add_constant(df[x_cols].astype(float))
        y = df[y_col].astype(float)
        if X.size == 0 or y.size == 0:
            summ = {"r2": None, "r2_adj": None, "n": 0}
            empty = pd.DataFrame(
                columns=["term", "coef", "stderr", "ci_low", "ci_high", "pval"]
            )
            return None, summ, empty
        model = sm.OLS(y, X, missing="drop").fit()
        summ = {
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "n": int(model.nobs),
        }
        coefs = model.params.to_frame("coef").join(model.bse.to_frame("stderr"))
        ci = model.conf_int()
        coefs["ci_low"] = ci[0]
        coefs["ci_high"] = ci[1]
        coefs["pval"] = model.pvalues
        return model, summ, coefs.reset_index().rename(columns={"index": "term"})
    except Exception as e:
        summ = {"r2": None, "r2_adj": None, "n": 0, "error": str(e)}
        empty = pd.DataFrame(
            columns=["term", "coef", "stderr", "ci_low", "ci_high", "pval"]
        )
        return None, summ, empty
