from collections import deque


def _check(requests, max_batch, token_budget):
    if token_budget < max_batch:
        raise ValueError("token_budget 至少要容纳 max_batch 个 decode token")
    for a, p, o in requests:
        if p > token_budget or o < 1:
            raise ValueError("prompt 超过 token_budget 或 output_len < 1")


def _result(requests, start, finish, tokens, running):
    steps = max(finish) + 1 if finish else 0
    tokens = tokens + [0] * (steps - len(tokens))
    running = running + [0] * (steps - len(running))
    total_out = sum(o for _, _, o in requests)
    return {"start": start, "finish": finish, "tokens_per_step": tokens[:steps],
            "running_per_step": running[:steps], "throughput": total_out / steps if steps else 0.0}


def _bump(lst, t, v):
    while len(lst) <= t:
        lst.append(0)
    lst[t] += v


def simulate_static(requests, max_batch, token_budget):
    _check(requests, max_batch, token_budget)
    n = len(requests)
    order = sorted(range(n), key=lambda i: (requests[i][0], i))
    start, finish = [None] * n, [None] * n
    tokens, running = [], []
    t, k = 0, 0
    while k < n:
        if requests[order[k]][0] > t:
            t = requests[order[k]][0]
            continue
        batch, used = [], 0
        while k < n and requests[order[k]][0] <= t and len(batch) < max_batch \
                and used + requests[order[k]][1] <= token_budget:
            i = order[k]
            batch.append(i)
            used += requests[i][1]
            k += 1
        dur = max(requests[i][2] for i in batch)
        for s in range(dur):
            alive = [i for i in batch if requests[i][2] > s]
            _bump(running, t + s, len(alive))
            _bump(tokens, t + s, used if s == 0 else len(alive))
        for i in batch:
            start[i] = t
            finish[i] = t + requests[i][2] - 1
        t += dur  # 整批结束后才能开下一批
    return _result(requests, start, finish, tokens, running)


def simulate_continuous(requests, max_batch, token_budget):
    _check(requests, max_batch, token_budget)
    n = len(requests)
    waiting = deque(sorted(range(n), key=lambda i: (requests[i][0], i)))
    start, finish = [None] * n, [None] * n
    gen = [0] * n
    run = []
    tokens, running = [], []
    t, done = 0, 0
    while done < n:
        budget = token_budget - len(run)          # 每个在跑的序列 decode 1 个 token
        new = []
        while waiting and requests[waiting[0]][0] <= t and len(run) + len(new) < max_batch \
                and requests[waiting[0]][1] <= budget:   # 严格 FCFS：队头放不下就停
            i = waiting.popleft()
            budget -= requests[i][1]
            start[i] = t
            new.append(i)
        tokens.append(token_budget - budget)
        running.append(len(run) + len(new))
        nxt = []
        for i in run + new:
            gen[i] += 1
            if gen[i] == requests[i][2]:
                finish[i] = t
                done += 1
            else:
                nxt.append(i)
        run = nxt
        t += 1
    return _result(requests, start, finish, tokens, running)
