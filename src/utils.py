import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def hist_dict_to_array(d: dict[int, int], max_bin: int = None) -> np.ndarray:

    if max_bin is None:
        max_bin = max(d.keys(), default=0)

    arr = np.zeros(max_bin + 1, dtype=int)
    for k, v in d.items():
        if 0 <= k <= max_bin:
            arr[k] = v
    return arr
