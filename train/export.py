import argparse
import os

import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("output")
    args = parser.parse_args()
    assert os.path.exists(args.file), "PyTorch checkpoint not found"

    output = args.output + ".npz" if not args.output.endswith(".npz") else args.output
    np_dict = {}
    state_dict = torch.load(args.file, map_location="cpu")
    for k, v in state_dict.items():
        v: torch.Tensor
        print(k, v.shape)
        # rust's ndarray doesn't support half precision
        t = v.contiguous().detach().cpu().numpy()
        np_dict[k] = t
    np.savez_compressed(output, **np_dict)


if __name__ == "__main__":
    main()
