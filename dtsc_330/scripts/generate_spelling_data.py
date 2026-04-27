import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "spelling_pairs.csv"


WORDS = [
    "hello","world","friend","address","receive","definitely","separate",
    "government","environment","argument","calendar","tomorrow","achieve",
    "accommodate","again","among","beginning","believe","business",
    "cemetery","coming","conscious","decision","disappear","embarrass",
    "existence","familiar","finally","foreign","forward","grateful",
    "happened","height","immediately","independent","interest",
    "knowledge","library","management","necessary","niece",
    "opportunity","possession","preferred","publicly","recommend",
    "relevant","remember","restaurant","rhythm","schedule",
    "sentence","similar","successful","tendency","tongue",
    "truly","until","vacuum","visible","whether","writing","yield"
]


def delete_char(word):
    if len(word) > 3:
        i = random.randint(0, len(word)-1)
        return word[:i] + word[i+1:]
    return word


def swap_chars(word):
    if len(word) > 3:
        i = random.randint(0, len(word)-2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return word


def replace_char(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    i = random.randint(0, len(word)-1)
    return word[:i] + random.choice(letters) + word[i+1:]


def insert_char(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    i = random.randint(0, len(word))
    return word[:i] + random.choice(letters) + word[i:]


def make_typo(word):
    operations = [delete_char, swap_chars, replace_char, insert_char]
    op = random.choice(operations)
    return op(word)


def main():
    data = []

    for _ in range(500):
        word = random.choice(WORDS)
        typo = make_typo(word)
        data.append((typo, word))

    df = pd.DataFrame(data, columns=["misspelled", "correct"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} spelling pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
