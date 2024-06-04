import argparse
import os
import random
from string import ascii_letters


def main(lang, seed):
    random.seed(seed)  # default to 3407, for reproducibility
    match lang:
        case "en":
            english()
        case "fr":
            prosodylab("fr")
        case "de":
            prosodylab("de")
        case _:
            print(
                f"Language code {lang} is not supported. Should be one of these: [`en`, `fr`, `de`]"
            )
    # divide them into train and validation
    # train: 90%, validation: 7.5%, test: 2.5%
    # save them into `data/{lang}-train.txt` and `data/{lang}-valid.txt`
    o_train = f"./data/{lang}-train.txt"
    o_valid = f"./data/{lang}-valid.txt"
    o_test = f"./data/{lang}-test.txt"
    f = open(f"./data/{lang}.txt")
    lines = f.readlines()
    f.close()
    train = []
    valid = []
    test = []
    length = len(lines)
    while len(valid) < length / 100 * 7.5:
        idx = random.randint(0, len(lines) - 1)
        valid.append(lines.pop(idx))
    while len(test) < length / 100 * 2.5:
        idx = random.randint(0, len(lines) - 1)
        test.append(lines.pop(idx))
    train = lines
    print(f"Train: {len(train)}\nValid: {len(valid)}\nTest: {len(test)}")
    f = open(o_train, "w")
    f.writelines(train)
    f.close()
    f = open(o_valid, "w")
    f.writelines(valid)
    f.close()
    f = open(o_test, "w")
    f.writelines(test)
    f.close()
    # delete the original file
    os.remove(f"./data/{lang}.txt")


def english():
    if not os.path.exists("./data"):
        os.mkdir("./data")
    file = open("./data/en.dict", "r")
    lines = file.readlines()
    file.close()
    res = []
    for line in lines:
        line = line.rstrip().lower()
        if line[:3] == ";;;":
            continue
        seg = line.split("#")
        if len(seg) > 1:
            r = seg[0]
            i = len(r) - 1
            while r[i] == " ":
                i -= 1
            line = r[: i + 1]
        filtered = False
        word = line.split("  ")[0]
        phoneme = line.split("  ")[1]
        # first remove the pos tagging surrended by `()`
        if "(" in word:
            word = word.split("(")[0]
        for char in word:
            # removes the only entry `threnos` that uses `-`
            if word == "threnos":
                filtered = True
                break
            # removes single character words
            if len(word) < 3:
                filtered = True
                break
            # removes non-ascii words
            if char not in ascii_letters + "'":
                filtered = True
        if not filtered:
            res.append(word + "  " + phoneme + "\n")
    o_file = open("./data/en.txt", "w")

    o_file.writelines(res)
    o_file.close()


def prosodylab(lang: str):
    f = open(f"./data/{lang}.dict")

    lines = f.readlines()

    f.close()

    f = open(f"./data/{lang}.txt", "w")

    alphabets = set()
    phonemes = set()

    for line in lines:
        line = line.rstrip().split(" ")
        word = line[0].lower()
        ph = line[1:]
        ph = [p.lower() for p in ph]

        for c in word:
            alphabets.add(c)
        for p in ph:
            phonemes.add(p)

        f.write(word)
        f.write("  ")
        f.write(" ".join(ph))
        f.write("\n")

    f.close()

    print(list(sorted(list(alphabets))))
    print(list(sorted(list(phonemes))))

    print(f"Length of alphabets: {len(alphabets)}\nLength of phonemes: {len(phonemes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", default="en")
    parser.add_argument("--seed", type=int, default=3407, required=False)
    args = parser.parse_args()
    main(args.lang, args.seed)
