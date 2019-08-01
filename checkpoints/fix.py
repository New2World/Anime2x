import os
import torch

fl = list(os.walk('.'))[0][2]
fl = [f for f in fl if f.endswith('.pt')]

for f in fl:
    n = int(f.split('-')[1])
    s = torch.load(f)
    # s['scheduler_state_dict']['last_epoch'] = n
    s['scheduler_state_dict']['step_size'] = 40
    torch.save(s, f)
    print(f'Ep.{n} - {s["scheduler_state_dict"]} - {[params["lr"] for params in s["optimizer_state_dict"]["param_groups"]]}')