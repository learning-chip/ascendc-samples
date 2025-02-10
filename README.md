# Ascend operator sample with torch interface in minimalist style

极简风格昇腾算子样例 + PyTorch调用

Advantages:
- Absolutely minimum and hackable software structure -- no complex cmake/bash/json wrapper as in [official AscendC project structure](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html)
- Easy-to-reason concurrent programming style using [binary semaphore APIs](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0269.html)

## Usage

Common environment setup

```bash
sudo docker pull quay.io/ascend/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10

sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $(pwd):/workdir \
    -w /workdir \
    --name custom_ops \
    quay.io/ascend/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 \
    /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

apt update && apt install -y gcc g++
pip install --upgrade pip setuptools wheel
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu  # x86 host
# pip install torch==2.4.0  # ARM host
pip install torch_npu==2.4.0
pip install pytest
```

Then follow the README in each subdirectory
