import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(series, window_size=30):
    """
    Convert a price series into overlapping windows.
    Returns normalized sequences and the fitted scaler.
    """
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    sequences = []
    for i in range(window_size, len(scaled)):
        sequences.append(scaled[i - window_size:i, 0])

    sequences = np.array(sequences)                    # (N, window_size)
    sequences = sequences[:, :, np.newaxis]            # (N, window_size, 1)
    return sequences, scaler


def label_regimes(sequences, threshold=0.02):
    """
    Auto-label each sequence by its net return:
    Bull  → net return > +threshold
    Bear  → net return < -threshold
    Crash → net return < -3x threshold (sharp drop)
    Neutral → otherwise
    """
    labels = []
    for seq in sequences:
        net_return = seq[-1, 0] - seq[0, 0]           # normalized price change
        if net_return < -3 * threshold:
            labels.append([0, 0, 0, 1])               # Crash
        elif net_return < -threshold:
            labels.append([0, 1, 0, 0])               # Bear
        elif net_return > threshold:
            labels.append([1, 0, 0, 0])               # Bull
        else:
            labels.append([0, 0, 1, 0])               # Neutral

    return np.array(labels)                            # (N, 4)