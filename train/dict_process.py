# this script is used to process the dictionary file
# and remove unwanted characters and words
import re
import argparse
import json
from os.path import join
from string import ascii_letters


# outputs 2 json files: `normal.en.json` and `homo.en.json`
def dict_process(path: str, out: str):
    # unlike in preprocess.py, we don't filter out single character words
    # or non-ascii words as they're essential
    # however, we do remove the only entry `threnos` that uses `-`
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    regex = re.compile(r"\(.*\)")
    normal = {}  # {word: phoneme}
    homo = {}  # {word: {pos: pos, phoneme: phoneme}}
    # pre-process
    pos_set = set()
    pre_lines = []
    for line in lines:
        line = line.rstrip().lower()
        if line[:3] == ";;;":
            continue
        seg = line.split("#")
        if len(seg) > 1:
            line = seg[0].rstrip()
        word = line.split("  ")[0]
        phoneme = line.split("  ")[1]
        phoneme = list(phoneme.split(" "))
        if word == "threnos":
            continue
        filtered = False
        for c in word:
            if c not in ascii_letters + "'()@":
                filtered = True
                break
        if filtered:
            continue
        pre_lines.append(line)
    # normal
    for line in pre_lines:
        word = line.split("  ")[0]
        phoneme = line.split("  ")[1]
        phoneme = list(phoneme.split(" "))
        match = regex.search(word)
        if not match:
            if word not in normal:
                normal[word] = phoneme
    # homo
    for line in pre_lines:
        word = line.split("  ")[0]
        phoneme = line.split("  ")[1]
        phoneme = list(phoneme.split(" "))
        match = regex.search(word)
        if match:
            pos = match.group()
            word = word.replace(pos, "")
            pos = pos[1:-1]
            pos_set.add(pos)
            if word not in homo:
                homo[word] = {}
            if pos not in homo[word]:
                homo[word][pos] = phoneme
            # use the first pos phoneme as the default normal phoneme
            if word not in normal:
                normal[word] = phoneme
    # some edge cases
    normal["the"] = ["dh", "ax"]

    homo_str = json.dumps(homo, separators=(",", ":"))
    normal_str = json.dumps(normal, separators=(",", ":"))

    with open(join(out, "homo.en.json"), "w") as f:
        f.write(homo_str)
    with open(join(out, "normal.en.json"), "w") as f:
        f.write(normal_str)
    print(f"{pos_set=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the dictionary file")
    parser.add_argument("out_dir", type=str, help="path to the output directory")
    args = parser.parse_args()
    dict_process(args.path, args.out_dir)
