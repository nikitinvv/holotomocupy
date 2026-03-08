import re
import sys
file = sys.argv[1]

def find_times(needle, rank):
    time_pat = re.compile(rf'{re.escape(needle)}\s*(\d+(?:\.\d+)?)\s*sec\b')

    times = []

    with open(file, "r", encoding="utf-8") as f:
        for lineno, text in enumerate(f, 1):
            if needle in text and f"[rank={rank}]" in text:
                m = time_pat.search(text)
                # print(text)
                if m:
                    times.append(float(m.group(1)))
    return times

def find_mem(rank):
        
    mem_pat = re.compile(rf'process memory\s*([0-9]+(?:\.[0-9]+)?)\s*GB\b')

    mem_gb = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if f"[rank={rank}]" in line:
                m = mem_pat.search(line)
                if m:
                    mem_gb.append(float(m.group(1)))
    return mem_gb

rank = 0
times = [0]*6
#
times[0] = sum(find_times('redist:',rank)[-2:])


# 3 fwd_tomo
times[1] = sum(find_times('fwd_tomo:',rank)[-1:])
# add 1 adj_tomo
times[1] += sum(find_times('inner:',rank)[-1:])


# 3 hessian
times[2] = sum(find_times('hessian:',rank)[-3:])


# gradient - 1inner -1redist
times[3] = sum(find_times('gradients:',rank)[-1:])-sum(find_times('inner:',rank)[-1:])-sum(find_times('redist:',rank)[-2:-1])

#min
times[4] = sum(find_times('min:',rank)[-1:])
# print(f'min {times[4]:.1f}')

total = sum(find_times('iter=1:',0)[-1:])
times[5] = total-sum(times)

mem = find_mem(0)[-1]


print(f'redist\t {times[0]:.1f}')
print(f'tomo\t {times[1]:.1f}')
print(f'hessians\t {times[2]:.1f}')
print(f'gradients\t {times[3]:.1f}')
print(f'other\t {times[5]:.1f}')
print(f'total\t {(total-times[4]):.1f}')
print(f'memory\t {mem:.1f}')


print(f'{times[0]:.1f}')
print(f'{times[1]:.1f}')
print(f'{times[2]:.1f}')
print(f'{times[3]:.1f}')
print(f'{times[5]:.1f}')
print(f'{(total-times[4]):.1f}')
print(f'{mem:.1f}')
