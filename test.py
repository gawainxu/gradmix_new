import fileinput

with fileinput.input('/home/zhi/projects/datasets/train_test_split.txt', inplace=True) as f:
    # Read the whole thing as one string
    for i, line in enumerate(f):
        # line.strip() removes the newline character (\n)
        if i % 4 == 0:
            line[-1] = "0"
        else:
            line[-1] = "1"