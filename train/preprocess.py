import argparse
import os
from string import ascii_letters


def main(lang):
    match lang:
        case "en":
            english()
            return
        case "fr":
            prosodylab("fr")
            return
        case "de":
            prosodylab("de")
            return
        case _:
            print(f"Language code {lang} is not supported. Should be one of these: [`en`, `fr`, `de`]")


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
        for char in word:
            # removes the only entry `threnos` that uses `-`
            if word == "threnos":
                filtered = True
                break
            # removes single character words
            if len(word) < 2:
                filtered = True
                break
            # removes non-ascii words
            # removes alternative pronounce
            if char not in ascii_letters + "'":
                filtered = True
                break
        if not filtered:
            res.append(line + "\n")
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
    args = parser.parse_args()
    main(args.lang)
