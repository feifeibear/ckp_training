### 测试checkpoint对模型训练的作用
使用
[TORCH.UTILS.CHECKPOINT](https://pytorch.org/docs/stable/checkpoint.html)
来减少activations的显存占用。


#### 测试方法
将一个4层全连接网络的前两层用checkpoint优化。

在项目的根目录

`export PYTHONPATH=$PYTHONPATH:$PWD`

运行

1. 查看checkpoint的内存使用
`tests/test_ckp.py --use_ckp`

2. 查看无checkpoint的内存使用
`tests/test_ckp.py`

3. 检查checkpoint结果是否正确
`tests/test_ckp.py --res_check`

#### 结果
不使用checkpoint优化(MA = Memory Allocation. CA = Cached Allocation)

MA 232.0 KB         Max_MA 245.5 KB         CA 2048.0 KB         Max_CA 2048 KB

使用checkpoint优化
MA 188.0 KB         Max_MA 201.5 KB         CA 2048.0 KB         Max_CA 2048 KB

显存使用有显著减少

#### 参考
[Megatron的checkpoint实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/random.py#L316)
