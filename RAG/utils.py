from typing import List


def load_txt_as_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()

    res = []
    for line in lines:
        res.append(line.strip())

    return res