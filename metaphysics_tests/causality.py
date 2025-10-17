import logging

import pandas as pd

logger = logging.getLogger(__name__)


def granger_pairwise(df, variables, maxlag=12, ic="aic"):
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
    except Exception as e:
        logger.exception("statsmodels not available: %s", e)
        return None
    model = VAR(df[variables].astype(float))
    try:
        res = model.fit(maxlags=maxlag, ic=ic)
    except Exception as e:
        logger.exception("VAR fit failed: %s", e)
        res = model.fit(maxlags=1)
    # pairwise Granger using Wald tests from fitted model
    p = len(variables)
    pairs = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            try:
                test = res.test_causality(variables[j], variables[i], kind="f")
                pairs.append(
                    {
                        "cause": variables[j],
                        "effect": variables[i],
                        "stat": float(test.test_statistic),
                        "pvalue": float(test.pvalue),
                        "lag": int(res.k_ar),
                    }
                )
            except Exception as e:
                logger.exception(
                    "causality test failed for %s->%s: %s",
                    variables[j],
                    variables[i],
                    e,
                )
    return pd.DataFrame(pairs)


def transfer_entropy_stub(df, variables):
    # try to import pyinform or dit; if not available return NA
    try:
        # not implemented here — placeholder
        return None
    except Exception:
        return None
