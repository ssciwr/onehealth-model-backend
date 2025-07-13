def add(x, y):
    return x + y


def multiply(x, y):
    return x * y


def subtract(x, y):
    return x - y


def affine(x, a=1, b=0):
    return a * x + b


def load_data(file_path):
    import pandas as pd

    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred while loading data: {e}")


def save_data(file_path, data):
    try:
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise ValueError(f"An error occurred while saving data: {e}")
