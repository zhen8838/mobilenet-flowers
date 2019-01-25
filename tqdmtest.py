from tqdm import trange, tqdm
from random import random, randint
from time import sleep

# '| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'


# def training(epoch: int, step_per_epoch: int):
#     for i in range(epoch):
#         with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix[0]}{postfix[1][loss]:>6.3f}',
#                   unit=' batch', postfix=['loss=', dict(loss=0)], ncols=80) as t:
#             for i in range(step_per_epoch):
#                 t.postfix[1]["loss"] = random()
# t.update()
def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', unit=' batch', ncols=80) as t:
            for i in range(step_per_epoch):
                t.set_postfix_str('loss={:^7.3f}'.format(random()))
                sleep(.1)
                t.update()


training(epoch=3, step_per_epoch=10)
