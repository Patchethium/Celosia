import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("output")
    args = parser.parse_args()
    assert os.path.exists(args.file)

    state_dict = torch.load(args.file, map_location="cpu")
    f = open(args.output, "wb")
    for k, v in state_dict.items():
        v: torch.Tensor
        print(k, v.shape)
        # save as fp16 to save space
        # on amepd full set, fp16 and f32 inference results are identical
        t = v.contiguous().view(-1).detach().cpu().half().numpy()
        f.write(memoryview(t))
    f.close()


if __name__ == "__main__":
    main()
