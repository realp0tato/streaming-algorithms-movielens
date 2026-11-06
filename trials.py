"""병렬 정확도 실험의 시행 함수. 별도 모듈이어야 worker마다 데이터 캐시가 유지된다.

정확도 시행만 메모리에 올린 목록을 재사용한다(결정적 결과라 속도 외 영향 없음).
시간과 메모리 측정은 run_experiments.py에서 파일을 한 줄씩 읽어 단일 프로세스로 수행한다.
"""
import io
import random
import sys
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

from sketches import BloomFilter, CountMinSketch, HyperLogLog, ReservoirL, ReservoirR

ZIP = Path(__file__).resolve().parent / "data" / "ml-1m.zip"
N_NEGATIVE = 1_000_000


def stream():
    """압축 파일에서 한 줄씩 읽어 (user, movie, rating, timestamp) 문자열을 내보낸다."""
    with zipfile.ZipFile(ZIP) as z, z.open("ml-1m/ratings.dat") as f:
        for line in io.TextIOWrapper(f, encoding="ascii"):
            yield line.rstrip("\n").split("::")


def pair_stream():
    for u, m, _, _ in stream():
        yield f"{u}:{m}"


def movie_stream():
    for _, m, _, _ in stream():
        yield m


def user_stream():
    for u, _, _, _ in stream():
        yield u


@lru_cache(maxsize=None)
def pairs():
    return list(pair_stream())


@lru_cache(maxsize=None)
def movies():
    return [sys.intern(m) for m in movie_stream()]


@lru_cache(maxsize=None)
def users():
    return [sys.intern(u) for u in user_stream()]


@lru_cache(maxsize=None)
def truth_counts():
    return Counter(movies())


@lru_cache(maxsize=1)
def negatives(seed):
    """실제 사용자 x 실제 영화 조합 중 평점 기록이 없는 쌍을 무작위로 뽑는다."""
    rng = random.Random(seed)
    known = set(pairs())
    us, ms = sorted(set(users())), sorted(set(movies()))
    out = []
    while len(out) < N_NEGATIVE:
        x = f"{rng.choice(us)}:{rng.choice(ms)}"
        if x not in known:
            out.append(x)
    return out


def trial_bloom(bits_per_item, k, seed):
    data = pairs()
    n = len(data)
    bf = BloomFilter(bits_per_item * n, k, seed)
    for x in data:
        bf.add(x)
    neg = negatives(seed)
    fp = sum(x in bf for x in neg)
    fn = sum(x not in bf for x in data)
    assert fn == 0, "Bloom 거짓 음성 발생"
    return dict(bits_per_item=bits_per_item, k=k, seed=seed, fpr=fp / len(neg), fp=fp, fn=fn,
                fill=bf.fill_ratio(), nbytes=bf.nbytes)


def trial_cms(w, d, seed, conservative):
    cms = CountMinSketch(w, d, seed, conservative)
    for x in movies():
        cms.add(x)
    c = truth_counts()
    keys = sorted(c, key=c.get, reverse=True)
    true = np.array([c[k] for k in keys], float)
    est = np.array([cms.query(k) for k in keys], float)
    err = est - true
    assert err.min() >= 0, "CMS 과소추정 발생"
    eps_n = np.e / w * cms.N
    rel = err / true
    low = true <= np.median(true)
    return dict(w=w, d=d, seed=seed, conservative=conservative, are=rel.mean(), mae=err.mean(), max_err=err.max(),
                are_top100=rel[:100].mean(), are_tail=rel[low].mean(), over_bound=(err > eps_n).mean(),
                exact_share=(err == 0).mean(), nbytes=cms.nbytes)


def trial_hll(p, target, seed):
    data = {"users": users, "movies": movies, "pairs": pairs}[target]()
    h = HyperLogLog(p, seed)
    for x in data:
        h.add(x)
    true = len(set(data))
    return dict(p=p, target=target, seed=seed, true=true, estimate=h.estimate(), nbytes=h.nbytes)


@lru_cache(maxsize=None)
def rating_arrays():
    r, m, u = [], [], []
    for a, b, c, _ in stream():
        u.append(int(a)); m.append(int(b)); r.append(int(c))
    return np.array(r), np.array(m), np.array(u)


def trial_reservoir(kind, k, seed):
    """kind: R, L, full(전체 데이터 기반 무작위 표본 = Ground Truth 분포), first(앞에서 k개, 나이브 대조군)"""
    rating, movie, user = rating_arrays()
    n = len(rating)
    if kind in ("R", "L"):
        res = (ReservoirR if kind == "R" else ReservoirL)(k, seed)
        for i in range(n):
            res.add(i)
        pos = np.array(res.sample)
    elif kind == "full":
        pos = np.array(random.Random(seed).sample(range(n), k))
    else:
        pos = np.arange(k)
    assert len(pos) == k and len(np.unique(pos)) == k

    p_true = np.bincount(rating, minlength=6)[1:] / n
    p_est = np.bincount(rating[pos], minlength=6)[1:] / k
    top = np.argsort(np.bincount(movie))[::-1][:100]
    c_true, c_est = np.bincount(movie), np.bincount(movie[pos], minlength=movie.max() + 1) * n / k
    return dict(kind=kind, k=k, seed=seed, mean_err=abs(rating[pos].mean() - rating.mean()),
                tv=.5 * np.abs(p_true - p_est).sum(), top100_are=(np.abs(c_est[top] - c_true[top]) / c_true[top]).mean(),
                users_in_sample=len(np.unique(user[pos])), ks_p=stats.kstest(pos / n, "uniform").pvalue,
                hist=np.histogram(pos, bins=20, range=(0, n))[0].tolist() if k == 10000 else None)
