## Sample 1 - Vector operator (silu activation) in AscendC

Compile and test:

```bash
# build operator as shared lib
bash ./compile.sh

# build torch extension
pip install .

# test run
pytest -v ./test_silu.py
```

Should get output:

```python
=========================================================================================== test session starts ============================================================================================
platform linux -- Python 3.10.15, pytest-8.3.4, pluggy-1.5.0 -- /usr/local/python3.10/bin/python3.10
cachedir: .pytest_cache
rootdir: /workdir/1_silu
collected 12 items                                                                                                                                                                                         

test_silu.py::test_silu_1d_input[10] PASSED                                                                                                                                                          [  8%]
test_silu.py::test_silu_1d_input[256] PASSED                                                                                                                                                         [ 16%]
test_silu.py::test_silu_1d_input[4011] PASSED                                                                                                                                                        [ 25%]
test_silu.py::test_silu_1d_input[18000] PASSED                                                                                                                                                       [ 33%]
test_silu.py::test_silu_2d_input[77-83] PASSED                                                                                                                                                       [ 41%]
test_silu.py::test_silu_2d_input[77-256] PASSED                                                                                                                                                      [ 50%]
test_silu.py::test_silu_2d_input[213-83] PASSED                                                                                                                                                      [ 58%]
test_silu.py::test_silu_2d_input[213-256] PASSED                                                                                                                                                     [ 66%]
test_silu.py::test_silu_2d_input[512-83] PASSED                                                                                                                                                      [ 75%]
test_silu.py::test_silu_2d_input[512-256] PASSED                                                                                                                                                     [ 83%]
test_silu.py::test_silu_2d_input[2048-83] PASSED                                                                                                                                                     [ 91%]
test_silu.py::test_silu_2d_input[2048-256] PASSED                                                                                                                                                    [100%]

============================================================================================ 12 passed in 4.26s ============================================================================================
```
