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
