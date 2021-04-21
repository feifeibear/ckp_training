from checkpoint import checkpoint
import logging
import torch
from tests.simple_net import SimpleCKPModel, SimpleModel, get_data_loader
from utils import see_memory_usage
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
import time
import argparse

parser = argparse.ArgumentParser(description='Checkpointing for Memory Saving.')
parser.add_argument('--use_ckp', dest='use_ckp', action = 'store_true',
                    help='an integer for the accumulator')
parser.add_argument('--res_check', dest='res_check', action = 'store_true',
                    help='check results correctness of checkpointing')


def test_simple_model(is_ckp: bool = False, is_fp16: bool = False):
    logging.info(f'test a simple model with checkpoit {is_ckp} FP16 {is_fp16}')

    hidden_dim = 40
    device = torch.device('cuda:0')

    if is_ckp:
        model = SimpleCKPModel(hidden_dim, empty_grad=False)
    else:
        model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    see_memory_usage(f"CKP {is_ckp} after model init", force = True)

    if is_fp16:
        model = FP16_Module(model)
        # model.half()

    data_loader = get_data_loader(
        model=model,
        total_samples=1000,
        hidden_dim=hidden_dim,
        device=device,
        data_type=torch.half if is_fp16 else torch.float)

    loss_res = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if is_fp16:
        optimizer = FP16_Optimizer(optimizer)

    start_time = time.time()
    for n, batch in enumerate(data_loader):

        loss = model(batch[0], batch[1])

        # if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        loss_res.append(loss.item())

        if is_fp16:
            optimizer.zero_grad(set_grads_to_None=True)
            optimizer.backward(loss, update_master_grads=False)
        else:
            optimizer.zero_grad()
            loss.backward()

        if is_fp16:
            # pass
            optimizer.update_master_grads()

        # chunk 0和 chunk 1还在compute状态
        optimizer.step()
        see_memory_usage(f"is_ckp {is_ckp} after step {n}", force = True)

        if n == 5: break

    elapse = time.time() - start_time
    logging.info(f"is_ckp {is_ckp} elapse {elapse}")
    return loss_res


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)
    
    args = parser.parse_args()
    use_ckp = args.use_ckp

    test_simple_model(is_ckp=use_ckp, is_fp16=True)

    # 检查结果正确性
    res_check = args.res_check
    print(use_ckp, res_check)
    if res_check:
        torch.manual_seed(0)
        loss_ref_list = test_simple_model(is_ckp=True, is_fp16=True)

        torch.cuda.empty_cache() 

        torch.manual_seed(0)
        loss_list = test_simple_model(is_ckp=False, is_fp16=True)

        print('ckp', loss_list)
        print('ref', loss_ref_list)
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref
