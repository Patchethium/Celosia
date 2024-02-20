import os

from string import ascii_letters


def main():
    if not os.path.exists("./data"):
        os.mkdir("./data")
    file = open("./data/amepd.txt", "r")
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


if __name__ == "__main__":
    main()
