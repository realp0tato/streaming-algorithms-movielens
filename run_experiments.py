"""MovieLens 1M 스트리밍 실험 전체. `python run_experiments.py` 로 results/*.csv 를 재생성한다.

순서: Ground Truth → 파라미터별 정확도(병렬, 결정적) → 시간·메모리(단일 프로세스, 파일 스트리밍).
"""
import gc
import hashlib
import json
import platform
import random
import statistics
import sys
import time
import tracemalloc
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

import trials
from sketches import BloomFilter, CountMinSketch, HyperLogLog, ReservoirL, ReservoirR
from trials import movie_stream, pair_stream, stream, user_stream

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
JOBS = 6  # 8GB 장비에서 worker별 데이터 캐시를 감안한 값
REPEATS = 7
SEEDS = {"bloom": 3, "cms": 5, "hll": 15, "reservoir": 30}


def ensure_data():
    if not trials.ZIP.exists():
        trials.ZIP.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, trials.ZIP)
    assert hashlib.sha256(trials.ZIP.read_bytes()).hexdigest() == SHA256, "데이터 파일이 예상과 다릅니다"


def parallel(fn, grid, name):
    print(name, len(grid), "runs", flush=True)
    t = time.time()
    rows = Parallel(n_jobs=JOBS)(delayed(fn)(*g) for g in grid)
    get_reusable_executor().shutdown(wait=True)  # worker의 데이터 캐시 해제
    print(f"  {time.time() - t:.0f}s", flush=True)
    return pd.DataFrame(rows)


def ground_truth():
    """한 줄씩 읽어 정확한 값을 계산한다. 원본 레코드는 저장하지 않고 집계만 남긴다."""
    print("ground truth", flush=True)
    n, ratings, ms, us, prs = 0, Counter(), Counter(), Counter(), set()
    last_u, sorted_user, ts_min, ts_max, inversions, last_t = 0, True, 1 << 62, 0, 0, 0
    sums = [0] * 10
    cnts = [0] * 10
    for u, m, r, t in stream():
        n += 1
        ratings[int(r)] += 1; ms[m] += 1; us[u] += 1; prs.add(f"{u}:{m}")
        sorted_user &= int(u) >= last_u
        last_u = int(u)
        t = int(t)
        ts_min, ts_max = min(ts_min, t), max(ts_max, t)
        inversions += t < last_t
        last_t = t
        b = min(9, (n - 1) // 100_021)
        sums[b] += int(r); cnts[b] += 1
    counts = np.array(sorted(ms.values(), reverse=True))
    with zipfile.ZipFile(trials.ZIP) as z:
        titles = dict(line.split("::")[:2] for line in z.read("ml-1m/movies.dat").decode("latin-1").splitlines())
    mean = sum(k * v for k, v in ratings.items()) / n
    max_u, max_m = max(map(int, us)), max(map(int, ms))
    gt = dict(n=n, users=len(us), movies=len(ms), pairs=len(prs), ratings={str(k): ratings[k] for k in sorted(ratings)},
              mean_rating=mean, sorted_by_user=bool(sorted_user), timestamp_inversions=inversions, ts_min=ts_min, ts_max=ts_max,
              top1_share=float(counts[0] / n), top100_share=float(counts[:100].sum() / n), top10pct_share=float(counts[:len(counts) // 10].sum() / n),
              median_movie_count=float(np.median(counts)), max_movie_count=int(counts[0]), min_movie_count=int(counts[-1]),
              movies_le_10=int((counts <= 10).sum()), top_movies=[[titles[m], c] for m, c in ms.most_common(5)],
              median_user_count=float(np.median(list(us.values()))), max_user_count=max(us.values()), min_user_count=min(us.values()),
              density=len(prs) / (len(us) * len(ms)), mean_by_decile=[s / c for s, c in zip(sums, cnts)],
              exact_bitmap_bytes=max_u * max_m // 8, first_1000_users=len({u for (u, *_), _ in zip(stream(), range(1000))}))
    (RES / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    pd.DataFrame(dict(count=counts)).to_csv(RES / "movie_rank_counts.csv", index=False)
    return gt


def stream_vs_load():
    """STEP 01: 한 줄씩 처리와 전체 적재의 메모리 차이."""
    tracemalloc.start()
    for _ in stream():
        pass
    _, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop(); gc.collect()
    tracemalloc.start()
    rows = list(stream())
    _, peak_load = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del rows; gc.collect()
    return dict(stream_peak=peak_stream, load_peak=peak_load)


class CounterAdd:
    """정확한 dict 카운터. 스케치와 호출 구조를 맞춘다."""

    def __init__(self):
        self.d = defaultdict(int)

    def add(self, x):
        self.d[x] += 1


class ListAdd:
    def __init__(self):
        self.items = []
        self.add = self.items.append


class SetAdd:
    def __init__(self):
        self.s = set()
        self.add = self.s.add


def configs(n):
    """(알고리즘, 라벨, 스트림, 생성자). 시간·메모리 측정의 공통 목록."""
    C = []
    for b, k in [(4, 3), (10, 1), (10, 4), (10, 7), (10, 14), (20, 14)]:
        C.append(("Bloom Filter", f"m/n={b}, k={k}", pair_stream, lambda b=b, k=k: BloomFilter(b * n, k, 1)))
    C.append(("정확(set)", "pair set", pair_stream, SetAdd))
    for w, d in [(2048, 1), (2048, 4), (2048, 16), (16384, 4)]:
        C.append(("Count-Min", f"w={w}, d={d}", movie_stream, lambda w=w, d=d: CountMinSketch(w, d, 1)))
    C.append(("Count-Min", "w=2048, d=4, CU", movie_stream, lambda: CountMinSketch(2048, 4, 1, True)))
    C.append(("정확(dict)", "movie dict", movie_stream, CounterAdd))
    for p in (6, 10, 14, 16):
        C.append(("HyperLogLog", f"p={p}", pair_stream, lambda p=p: HyperLogLog(p, 1)))
    for cls, name in ((ReservoirR, "R"), (ReservoirL, "L")):
        for k in (1000, 100000):
            C.append(("Reservoir", f"{name}, k={k}", pair_stream, lambda cls=cls, k=k: cls(k, 1)))
    C.append(("정확(list)", "전체 저장", pair_stream, ListAdd))
    return C


def timing(n):
    """매 반복마다 전체 설정을 섞어 한 번씩 측정한다. 순서를 섞어야 발열·메모리 회수 같은 시간대 효과가 특정 설정에 쏠리지 않는다."""
    print("timing", flush=True)
    time.sleep(15)  # 병렬 worker 종료 후 메모리 회수 대기
    tasks = [("기준", "pair_stream", pair_stream, None), ("기준", "movie_stream", movie_stream, None)] + configs(n)
    times = [[] for _ in tasks]
    rng = random.Random(0)
    for rep in range(REPEATS):
        for i in rng.sample(range(len(tasks)), len(tasks)):
            _, _, s, make = tasks[i]
            gc.collect()
            if make is None:
                t = time.perf_counter()
                for _x in s():
                    pass
            else:
                obj = make()
                add = obj.add
                t = time.perf_counter()
                for x in s():
                    add(x)
            times[i].append(time.perf_counter() - t)
            if make is not None:
                del obj, add
        print(f"  repeat {rep + 1}/{REPEATS}", flush=True)
    base = {tasks[i][1]: statistics.median(times[i]) for i in (0, 1)}
    rows = []
    for (algo, label, s, _), ts in zip(tasks[2:], times[2:]):
        total = statistics.median(ts)
        net = total - base[s.__name__]
        rows.append(dict(algo=algo, label=label, stream=s.__name__, total_s=total, min_s=min(ts), max_s=max(ts),
                         net_s=net, base_s=base[s.__name__], rate=n / total, us_per_item=net / n * 1e6))
    pd.DataFrame(rows).to_csv(RES / "timing.csv", index=False)

    # 조회 속도. 음성 조회는 Bloom의 조기 종료 덕분에 양성보다 빠르다.
    data = list(pair_stream())
    neg = trials.negatives(1)[:200000]
    pos = data[:200000]
    q = []
    for b, k in [(10, 7), (20, 14)]:
        bf = BloomFilter(b * n, k, 1)
        for x in data:
            bf.add(x)
        for name, items in (("양성", pos), ("음성", neg)):
            t = time.perf_counter()
            for x in items:
                x in bf
            q.append(dict(what=f"Bloom m/n={b}, k={k}, {name}", rate=len(items) / (time.perf_counter() - t)))
    s = set(data)
    t = time.perf_counter()
    for x in neg:
        x in s
    q.append(dict(what="정확 set, 음성", rate=len(neg) / (time.perf_counter() - t)))
    cms = CountMinSketch(2048, 4, 1)
    for x in movie_stream():
        cms.add(x)
    keys = sorted(set(movie_stream())) * 50
    t = time.perf_counter()
    for x in keys:
        cms.query(x)
    q.append(dict(what="Count-Min w=2048, d=4", rate=len(keys) / (time.perf_counter() - t)))
    h = HyperLogLog(14, 1)
    for x in data[:100000]:
        h.add(x)
    t = time.perf_counter()
    for _ in range(50):
        h.estimate()
    q.append(dict(what="HyperLogLog p=14 estimate()", rate=50 / (time.perf_counter() - t)))
    pd.DataFrame(q).to_csv(RES / "query_rate.csv", index=False)


def memory(n):
    """tracemalloc은 처리 시간을 늘리므로 시간 측정과 분리한다. 측정값은 Python 힙 기준."""
    print("memory", flush=True)
    base = {}
    for s in (pair_stream, movie_stream):
        tracemalloc.start()
        for _ in s():
            pass
        base[s.__name__] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop(); gc.collect()
    rows = []
    for algo, label, s, make in configs(n):
        gc.collect()
        tracemalloc.start()
        obj = make()
        add = obj.add
        for x in s():
            add(x)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(algo=algo, label=label, retained=cur, peak=peak, peak_net=peak - base[s.__name__],
                         payload=getattr(obj, "nbytes", None)))
        print(f"  {algo} {label}: {cur / 1e6:.2f} MB", flush=True)
        del obj, add
    pd.DataFrame(rows).to_csv(RES / "memory.csv", index=False)


def main():
    RES.mkdir(exist_ok=True)
    ensure_data()
    if "--timing-only" in sys.argv:  # 시간 측정만 다시 할 때
        timing(json.loads((RES / "ground_truth.json").read_text())["n"])
        return
    gt = ground_truth()
    sv = stream_vs_load()
    (RES / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), machine=platform.machine(), **sv,
        numpy=np.__version__, pandas=pd.__version__, repeats=REPEATS, seeds=SEEDS), indent=1))

    # 정확도: 시드 고정이라 재실행해도 동일하다.
    B = [(b, k, s) for b in (2, 4, 6, 8, 10, 12, 16, 20) for k in (1, 2, 3, 4, 5, 7, 10, 14) for s in range(SEEDS["bloom"])]
    parallel(trials.trial_bloom, B, "bloom").to_csv(RES / "bloom.csv", index=False)

    W = [2 ** i for i in range(6, 16)]
    C = [(w, d, s, False) for w in W for d in (1, 2, 4, 8, 16) for s in range(SEEDS["cms"])]
    C += [(w, d, s, True) for w in W for d in (2, 4) for s in range(SEEDS["cms"])]
    parallel(trials.trial_cms, C, "cms").to_csv(RES / "cms.csv", index=False)

    H = [(p, t, s) for p in range(4, 17) for t in ("users", "movies", "pairs") for s in range(SEEDS["hll"])]
    parallel(trials.trial_hll, H, "hll").to_csv(RES / "hll.csv", index=False)

    K = (100, 500, 1000, 5000, 10000, 50000, 100000)
    R = [(kind, k, s) for kind in ("R", "L", "full") for k in K for s in range(SEEDS["reservoir"])]
    R += [("first", k, 0) for k in K]
    r = parallel(trials.trial_reservoir, R, "reservoir")
    hist = r[r["hist"].notna()].groupby("kind")["hist"].apply(lambda h: np.sum(h.tolist(), axis=0).tolist())
    (RES / "reservoir_hist.json").write_text(hist.to_json(force_ascii=False))
    r.drop(columns="hist").to_csv(RES / "reservoir.csv", index=False)

    # 시간·메모리: 병렬 작업이 모두 끝난 뒤 단일 프로세스로 측정
    timing(gt["n"])
    memory(gt["n"])
    print("done", flush=True)


if __name__ == "__main__":
    random.seed(0)
    main()
