def classify(df):
    """
    Classify study results based on OpenLaws validation criteria.

    Args:
        df: DataFrame with delta_cci and delta_hazard columns

    Returns:
        str: "VALIDATED" or "UNDER_REVIEW"
    """
    pass_rate = (df["delta_cci"].mean() >= 0.03) and (
        df["delta_hazard"].mean() <= -0.01
    )
    return "VALIDATED" if pass_rate else "UNDER_REVIEW"
