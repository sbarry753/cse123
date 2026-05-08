# Makes random input sample for the known answer test
# MCU compares its CNN output with the expected output computed on the host
import numpy as np
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dimensions", type=tuple, default=None, help="Expected input dimensions.")
    p.add_argument("--output", type=str, default="sample_MAXGuitarPiano.npy")
    return p.parse_args()


def main():
    args = parse_args()
    if args is None or args.input_dimensions is None:
        input_dimensions = (1, 513, 5)

    a = np.random.randint(-128, 127, size=input_dimensions, dtype=np.int64)
    np.save(args.output, a, allow_pickle=False)

if __name__ == "__main__":
    main()