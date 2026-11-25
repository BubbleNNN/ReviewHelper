import numpy as np


def ring_allreduce(buffers):
    p = len(buffers)
    # chunks[r][c]：rank r 的第 c 块（拷贝，不改输入）
    chunks = [[c.astype(np.float64, copy=True) for c in np.array_split(np.asarray(b), p)] for b in buffers]
    log = []
    step = 0
    # reduce-scatter：第 s 步 rank r 把块 (r - s) % p 发给 r+1，接收方累加
    for s in range(p - 1):
        sends = [(r, (r + 1) % p, (r - s) % p, chunks[r][(r - s) % p].copy()) for r in range(p)]
        for src, dst, cid, data in sends:
            chunks[dst][cid] += data
            log.append((step, src, dst, cid))
        step += 1
    # all-gather：第 s 步 rank r 把块 (r + 1 - s) % p 发给 r+1，接收方覆盖
    for s in range(p - 1):
        sends = [(r, (r + 1) % p, (r + 1 - s) % p, chunks[r][(r + 1 - s) % p].copy()) for r in range(p)]
        for src, dst, cid, data in sends:
            chunks[dst][cid] = data
            log.append((step, src, dst, cid))
        step += 1
    results = [np.concatenate(cs) if cs else np.zeros(0) for cs in chunks]
    return results, log


def allreduce_time(n_bytes, p, alpha, beta):
    if p <= 1:
        return 0.0
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * n_bytes * beta
