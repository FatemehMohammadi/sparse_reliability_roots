# Triangle examples for “A unit-disk transition for reliability roots of sparse random simplicial complexes”

To reproduce Table 1, run:

```sh
python -m pip install numpy==2.3.5
python sparse_examples.py
```

The script writes CSV to standard output. With seed 1905, it samples 45 labels first and then 75, using `c = 1, 4, 8` for each size. The last three rows give Table 1. Running only `--n 75` gives different samples because it skips the first three draws.
