import torch
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("output")
    args = parser.parse_args()
    assert os.path.exists(args.file)

    state_dict = torch.load(args.file, map_location="cpu")
    f = open(args.output, "wb")
    for k, v in state_dict.items():
        print(k, v.shape)
        t = v.contiguous().view(-1).detach().cpu().numpy()
        f.write(memoryview(t))
    f.close()


if __name__ == "__main__":
    main()
