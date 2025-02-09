# `bisheng` CLI: https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/BishengCompiler/atlas_bisheng_10_0003.html
# Adapted from: https://gitee.com/ascend/mstt/tree/master/sample/pytorch_adapter

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --cce-soc-version=Ascend910B4 --cce-soc-core-type=VecCore \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -o libsilu_ascendc.so silu.cpp
