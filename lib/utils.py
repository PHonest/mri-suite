from typing import Any

def flatten(l: list[Any]) -> list[Any]:
    return [item for sublist in l for item in sublist]


def remove_dupes(l: list[tuple], key=0) -> list[tuple]:
    seen = set()
    result = []
    for item in l:
        if item[key] not in seen:
            seen.add(item[key])
            result.append(item)
    return result

