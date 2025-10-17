import hashlib
import pathlib


def save(df, path):
    """
    Save DataFrame to CSV with SHA256 integrity hash.

    Args:
        df: DataFrame to save
        path: Output path for CSV file
    """
    p = pathlib.Path(path)
    df.to_csv(p, index=False)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print("SHA256:", h)
