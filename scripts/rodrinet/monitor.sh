#!/bin/bash
# Training progress monitor
cd /home/berk/VS_Projects/ResearchLab
python3 -c "
import os, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = sorted([d for d in os.listdir('logs') if d.startswith('PPO')], reverse=True)
if not log_dirs:
    print('No logs found yet')
    exit()

ea = EventAccumulator(f'logs/{log_dirs[0]}')
ea.Reload()

rew   = ea.Scalars('rollout/ep_rew_mean')
fps   = ea.Scalars('time/fps')
steps = ea.Scalars('rollout/ep_len_mean')

if rew:
    last = rew[-1]
    total = 2_000_000
    pct = last.step / total * 100
    remaining = (total - last.step) / fps[-1].value / 60 if fps else 0
    print(f'Step: {last.step:>8,} / {total:,}  ({pct:.1f}%)')
    print(f'Mean reward: {last.value:.1f}')
    print(f'FPS:         {fps[-1].value:.0f}')
    print(f'Est. remaining: {remaining:.0f} min')
" 2>/dev/null
