"""results/ 로 그림과 보고서를 만들고 Chrome 헤드리스로 PDF를 출력한다.
수치는 전부 결과 파일에서 읽는다. GitHub 주소는 인자로 받는다: python build_report.py <url>
"""
import base64
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
OUT = ROOT / "report" / "스트리밍알고리즘_보고서_2243175_윤승희.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
GITHUB = f'<a href="{sys.argv[1]}">{sys.argv[1]}</a>' if len(sys.argv) > 1 else "(GitHub 주소 입력 전)"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Malgun Gothic", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "mathtext.fontset": "dejavusans", "mathtext.default": "rm", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

GT = json.loads((RES / "ground_truth.json").read_text())
ENV = json.loads((RES / "environment.json").read_text())
HIST = json.loads((RES / "reservoir_hist.json").read_text())
N = GT["n"]

bloom = pd.read_csv(RES / "bloom.csv").groupby(["bits_per_item", "k"]).agg(fpr=("fpr", "mean"), fill=("fill", "mean"), fn=("fn", "sum"), fp=("fp", "sum")).reset_index()
bloom["theory"] = (1 - np.exp(-bloom.k / bloom.bits_per_item)) ** bloom.k
cms_raw = pd.read_csv(RES / "cms.csv")
cms = cms_raw.groupby(["conservative", "w", "d"]).mean(numeric_only=True).reset_index()
hll_raw = pd.read_csv(RES / "hll.csv")
hll_raw["rel"] = (hll_raw.estimate - hll_raw.true) / hll_raw.true
hll = hll_raw.groupby(["target", "p"]).rel.agg(rmse=lambda x: np.sqrt((x ** 2).mean()), bias="mean").reset_index()
res_raw = pd.read_csv(RES / "reservoir.csv")
res = res_raw.groupby(["kind", "k"]).mean(numeric_only=True).reset_index()
TM = pd.read_csv(RES / "timing.csv")
MEM = pd.read_csv(RES / "memory.csv")
Q = pd.read_csv(RES / "query_rate.csv").set_index("what").rate


def bf(b, k, col="fpr"):
    return bloom[(bloom.bits_per_item == b) & (bloom.k == k)][col].iloc[0]


def cm(w, d, col="are", cu=False):
    return cms[(cms.w == w) & (cms.d == d) & (cms.conservative == cu)][col].iloc[0]


def hl(p, target="pairs", col="rmse"):
    return hll[(hll.p == p) & (hll.target == target)][col].iloc[0]


def rs(kind, k, col):
    return res[(res.kind == kind) & (res.k == k)][col].iloc[0]


def tm(label, col="us_per_item"):
    return TM[TM.label == label][col].iloc[0]


def mem(label, col="retained"):
    return MEM[MEM.label == label][col].iloc[0]


def size(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.1f} KB"


def rate(x):
    return f"{x / 1e3:,.0f}K/s" if x >= 1e3 else f"{x:,.0f}/s"


def table(head, rows, cls="", bold=None):
    bold = bold or set()
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td{' class=b' if (i, j) in bold else ''}>{c}</td>" for j, c in enumerate(r)) + "</tr>" for i, r in enumerate(rows))
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def fig(name, cap, width=100):
    b = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    return f'<figure><img style="width:{width}%" src="data:image/png;base64,{b}"><figcaption>{cap}</figcaption></figure>'


def decade(x, _):
    return f"$10^{{{round(np.log10(x))}}}$" if x > 0 else ""


def save(f, name):
    # 기본 로그 눈금은 본문 글꼴로 음수 부호를 그려 한글 글꼴에서 깨진다
    for a in f.axes:
        for axis in (a.xaxis, a.yaxis):
            if axis.get_scale() == "log" and axis.get_transform().base == 10:  # depth 축은 2진 눈금 유지
                axis.set_major_formatter(FuncFormatter(decade))
                axis.set_minor_formatter(NullFormatter())
    f.savefig(FIG / f"{name}.png")
    plt.close(f)


# ── 그림 ────────────────────────────────────────────────────────────────────
f, ax = plt.subplots(1, 3, figsize=(10, 2.9))
vals = [GT["ratings"][str(i)] / N * 100 for i in range(1, 6)]
ax[0].bar(range(1, 6), vals, color=BLUE, width=.6)
for i, v_ in enumerate(vals, 1):
    ax[0].text(i, v_ + .7, f"{v_:.1f}", ha="center", color=INK, fontsize=9)
ax[0].set(title="평점 분포", xlabel="평점", ylabel="비율 (%)", ylim=(0, 42))
ax[0].grid(axis="x", visible=False)
counts = pd.read_csv(RES / "movie_rank_counts.csv")["count"].to_numpy()
ax[1].loglog(range(1, len(counts) + 1), counts, color=BLUE, lw=2)
ax[1].set(title="영화별 평점 수 (순위-빈도)", xlabel="영화 순위", ylabel="평점 수")
ax[2].plot(range(1, 11), GT["mean_by_decile"], "-o", color=BLUE, lw=2, ms=5)
ax[2].axhline(GT["mean_rating"], color=MUTED, ls=":", lw=1)
ax[2].set(title="스트림 구간별 평균 평점", xlabel="스트림 위치 (10등분)", ylabel="평균 평점", ylim=(3.4, 3.8))
f.tight_layout(); save(f, "eda")

f, ax = plt.subplots(1, 2, figsize=(10, 3.3))
for col, k in zip(SERIES, (1, 3, 7, 14)):
    q = bloom[bloom.k == k]
    ax[0].semilogy(q.bits_per_item, q.fpr, "-o", color=col, lw=2, ms=5, label=f"k={k}")
    ax[0].semilogy(q.bits_per_item, q.theory, ":", color=col, lw=1)
ax[0].set(title="메모리 증가에 따른 FPR (점선: 이론값)", xlabel="원소당 비트 수 m/n", ylabel="False Positive Rate (로그)")
ax[0].legend()
for col, b in zip(SERIES, (4, 8, 12, 16, 20)):
    q = bloom[bloom.bits_per_item == b]
    ax[1].semilogy(q.k, q.fpr, "-o", color=col, lw=2, ms=5, label=f"m/n={b}")
ax[1].set(title="해시 수 k에 따른 FPR", xlabel="해시 함수 개수 k", ylabel="False Positive Rate (로그)")
ax[1].legend(ncol=2)
f.tight_layout(); save(f, "bloom")

std = cms[~cms.conservative]
f, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
for col, d in zip(SERIES, (1, 2, 4, 8, 16)):
    q = std[(std.d == d) & (std.are > 0)]
    ax[0].loglog(q.nbytes / 1e3, q.are, "-o", color=col, lw=1.8, ms=4, label=f"d={d}")
ax[0].set(title="메모리와 평균 상대오차", xlabel="스케치 크기 (KB)", ylabel="ARE (로그)")
ax[0].legend(ncol=2, fontsize=8.5)
for col, budget in zip(SERIES, (2 ** 12, 2 ** 14)):
    q = std[std.w * std.d == budget].sort_values("d")
    ax[1].semilogy(q.d, q.are, "-o", color=col, lw=2, ms=5, label=f"카운터 {budget:,}개 ({budget * 4 / 1e3:.0f} KB)")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
ax[1].set(title="같은 메모리에서 depth 변경", xlabel="depth d (w = 예산/d)", ylabel="ARE (로그)")
ax[1].legend(fontsize=8.5)
q = std[std.d == 4]
ax[2].loglog(q.w, q.are_tail.where(q.are_tail > 0), "-o", color=ORANGE, lw=2, ms=5, label="하위 50% 영화")
ax[2].loglog(q.w, q.are_top100.where(q.are_top100 > 0), "-o", color=BLUE, lw=2, ms=5, label="상위 100 영화")
ax[2].set(title="인기도별 오차 (d=4)", xlabel="width w", ylabel="ARE (로그)")
ax[2].legend(fontsize=8.5)
f.tight_layout(); save(f, "cms")

f, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ps = np.arange(4, 17)
for col, (t, lab) in zip(SERIES, (("pairs", f"쌍 ({GT['pairs']:,})"), ("users", f"사용자 ({GT['users']:,})"), ("movies", f"영화 ({GT['movies']:,})"))):
    q = hll[hll.target == t]
    ax[0].semilogy(q.p, q.rmse, "-o", color=col, lw=2, ms=4.5, label=lab)
ax[0].semilogy(ps, 1.04 / np.sqrt(2.0 ** ps), ":", color=INK, lw=1.3, label="이론 1.04/√m")
ax[0].set(title="정밀도 p와 상대오차 RMSE", xlabel="p (레지스터 2^p개, 각 1바이트)", ylabel="RMSE (로그)")
ax[0].legend(fontsize=8.5)
q = hll_raw[hll_raw.target == "pairs"]
ax[1].boxplot([q[q.p == p].rel * 100 for p in ps], positions=ps, widths=.55, patch_artist=True, showfliers=True,
              boxprops=dict(facecolor=BLUE, alpha=.6, linewidth=.8), medianprops=dict(color=INK), whiskerprops=dict(linewidth=.8),
              capprops=dict(linewidth=.8), flierprops=dict(markersize=2.5))
ax[1].axhline(0, color=MUTED, lw=.8)
ax[1].set(title="쌍 스트림 추정 오차의 분포 (시드 15개)", xlabel="p", ylabel="상대오차 (%)")
f.tight_layout(); save(f, "hll")

f, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
for col, (kind, lab) in zip((BLUE, ORANGE, AQUA, YELLOW), (("R", "Algorithm R"), ("L", "Algorithm L"), ("full", "전체 데이터 무작위"), ("first", "앞에서 k개"))):
    q = res[res.kind == kind]
    ax[0].loglog(q.k, q.tv, "-o", color=col, lw=2, ms=4.5, label=lab)
ax[0].set(title="평점 분포의 총변동거리", xlabel="표본 크기 k", ylabel="TV 거리 (로그)")
ax[0].legend(fontsize=8.5)
x = np.arange(20)
ax[1].bar(x - .2, np.array(HIST["R"]) / 30, width=.4, color=BLUE, label="Algorithm R")
ax[1].bar(x + .2, np.array(HIST["L"]) / 30, width=.4, color=ORANGE, label="Algorithm L")
ax[1].set(title="표본의 스트림 위치 (k=10,000)", xlabel="스트림 위치 (20등분)", ylabel="시행당 표본 수", ylim=(0, 720))
ax[1].grid(axis="x", visible=False)
ax[1].legend(fontsize=8.5, loc="upper center", ncol=2)
for col, (kind, lab) in zip((BLUE, YELLOW), (("R", "Algorithm R"), ("first", "앞에서 k개"))):
    q = res[res.kind == kind]
    ax[2].loglog(q.k, q.users_in_sample, "-o", color=col, lw=2, ms=4.5, label=lab)
ax[2].set(title="표본에 포함된 사용자 수", xlabel="표본 크기 k", ylabel="고유 사용자 (로그)")
ax[2].legend(fontsize=8.5)
f.tight_layout(); save(f, "reservoir")

# 알고리즘별 정확도-메모리 곡선
best_bf = bloom.loc[bloom.groupby("bits_per_item").fpr.idxmin()]
f, ax = plt.subplots(figsize=(7.4, 3.8))
ax.loglog(best_bf.bits_per_item * N / 8 / 1e3, best_bf.fpr, "-o", color=BLUE, lw=2, ms=5, label="Bloom Filter: FPR (최적 k)")
q = std[(std.d == 4) & (std.are > 0)]
ax.loglog(q.nbytes / 1e3, q.are_top100.where(q.are_top100 > 0), "-o", color=ORANGE, lw=2, ms=5, label="Count-Min: 상위 100 ARE (d=4)")
q = hll[hll.target == "pairs"]
ax.loglog(2.0 ** q.p / 1e3, q.rmse, "-o", color=AQUA, lw=2, ms=5, label="HyperLogLog: 상대오차 RMSE")
mem_per = mem("R, k=100000") / 100000
q = res[res.kind == "R"]
ax.loglog(q.k * mem_per / 1e3, q.tv, "-o", color=YELLOW, lw=2, ms=5, label="Reservoir: 평점 분포 TV 거리")
ax.set(title="메모리와 오차 (지표가 서로 달라 기울기만 비교)", xlabel="자료구조 크기 (KB, 로그)", ylabel="각 알고리즘의 오차 지표 (로그)")
ax.legend(fontsize=8.5)
save(f, "tradeoff")

rep = [("m/n=10, k=7", "Bloom m/n=10, k=7", BLUE), ("pair set", "정확 set (쌍)", MUTED), ("w=2048, d=4", "CMS w=2048, d=4", ORANGE), ("w=16384, d=4", "CMS w=16384, d=4", ORANGE),
       ("movie dict", "정확 dict (영화)", MUTED), ("p=14", "HLL p=14", AQUA), ("R, k=1000", "Reservoir R k=1,000", YELLOW), ("R, k=100000", "Reservoir R k=100,000", YELLOW), ("전체 저장", "전체 list 저장", MUTED)]
f, ax = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=True)
ys = np.arange(len(rep))[::-1]
ax[0].barh(ys, [mem(l) / 1e6 for l, _, _ in rep], color=[c for *_, c in rep], height=.6)
ax[0].set_xscale("log")
ax[0].set_yticks(ys, [n for _, n, _ in rep])
for y, (l, _, _) in zip(ys, rep):
    ax[0].text(mem(l) / 1e6 * 1.2, y, size(mem(l)), va="center", fontsize=8.5, color=INK)
ax[0].set(title="메모리 사용량 (tracemalloc, 로그)", xlabel="MB", xlim=(.005, 2000))
ax[1].barh(ys, [tm(l, "rate") / 1e6 for l, _, _ in rep], color=[c for *_, c in rep], height=.6)
for y, (l, _, _) in zip(ys, rep):
    ax[1].text(tm(l, "rate") / 1e6 + .03, y, f"{tm(l, 'rate') / 1e6:.2f}M/s", va="center", fontsize=8.5, color=INK)
ax[1].set(title="처리량 (파일 읽기·파싱 포함)", xlabel="초당 레코드 (백만)", xlim=(0, 2.4))
for a in ax:
    a.grid(axis="y", visible=False)
f.tight_layout(); save(f, "mem_time")

# ── 본문 값 ────────────────────────────────────────────────────────────────
reliable = bloom[bloom.fp >= 300]
ratio_lo, ratio_hi = (reliable.fpr / reliable.theory).min(), (reliable.fpr / reliable.theory).max()
d14 = [cm(2 ** 14 // d, d) for d in (1, 2, 4, 8, 16)]
d12 = [cm(2 ** 12 // d, d) for d in (1, 2, 4, 8, 16)]
assert np.argmin(d14) in (1, 2, 3)  # 본문의 'U자' 서술이 결과와 일치하는지 확인
fit_hll = np.polyfit(np.log(2.0 ** hll[hll.target == "pairs"].p), np.log(hll[hll.target == "pairs"].rmse), 1)[0]
q = res[res.kind == "R"]
fit_res = np.polyfit(np.log(q.k), np.log(q.tv), 1)[0]
q = std[(std.d == 1)]
fit_cms = np.polyfit(np.log(q.w), np.log(q.are), 1)[0]
over = cms[~cms.conservative].assign(delta=lambda x: np.exp(-x.d))
bound_ok = bool((over.over_bound <= over.delta).all())
assert bound_ok  # '보장 범위 내' 서술 검증
assert bloom.fn.sum() == 0
ks = res_raw[res_raw.kind.isin(["R", "L"])].ks_p
ks_rate = (ks < .05).mean()
from scipy import stats  # noqa: E402
chi = {k: stats.chisquare(HIST[k]).pvalue for k in ("R", "L", "full")}

bf_rows = []
KS = (1, 2, 3, 4, 5, 7, 10, 14)
bf_bold = set()
for i, b in enumerate((2, 4, 6, 8, 10, 12, 16, 20)):
    vals = [bf(b, k) for k in KS]
    bf_rows.append([f"{b}", size(b * N / 8)] + [f"{x:.5f}" for x in vals])
    bf_bold.add((i, 2 + int(np.argmin(vals))))
W = [2 ** i for i in range(6, 16)]
cm_rows, cm_bold = [], set()
for i, w in enumerate(W):
    cm_rows.append([f"{w:,}"] + [f"{cm(w, d):.3f}" for d in (1, 2, 4, 8, 16)] + [size(w * 4 * 4)])
hl_rows = []
for p in (4, 6, 8, 10, 12, 14, 16):
    hl_rows.append([p, f"{2 ** p:,}", size(2 ** p), f"{1.04 / 2 ** (p / 2):.2%}"] + [f"{hl(p, t):.2%}" for t in ("pairs", "users", "movies")])
rs_rows = []
for k in (100, 1000, 10000, 100000):
    rs_rows.append([f"{k:,}"] + [f"{rs(a, k, 'tv'):.4f}" for a in ("R", "L", "full", "first")] + [f"{rs('R', k, 'mean_err'):.4f}", f"{rs('first', k, 'mean_err'):.4f}", f"{rs('R', k, 'users_in_sample'):,.0f}", f"{rs('first', k, 'users_in_sample'):,.0f}"])
m_rows = []
for r in MEM.itertuples():
    m_rows.append([r.algo, r.label, size(r.retained), size(r.peak_net), "-" if pd.isna(r.payload) else size(r.payload)])
t_rows = []
for r in TM.itertuples():
    t_rows.append([r.algo, r.label, f"{r.total_s:.2f}", f"{r.min_s:.2f}~{r.max_s:.2f}", rate(r.rate), f"{r.us_per_item:.2f}"])
q_rows = [[k, rate(v)] for k, v in Q.items()]
summary = [
    ["Bloom Filter", "m/n=10, k=7", f"FPR {bf(10, 7):.2%}", size(mem("m/n=10, k=7")), size(mem("pair set")), f"{mem('pair set') / mem('m/n=10, k=7'):,.0f}배", rate(tm("m/n=10, k=7", "rate"))],
    ["Count-Min", "w=2048, d=4", f"상위 100 ARE {cm(2048, 4, 'are_top100'):.1%}, 전체 ARE {cm(2048, 4):.2f}", size(mem("w=2048, d=4")), size(mem("movie dict")), f"{mem('movie dict') / mem('w=2048, d=4'):,.0f}배", rate(tm("w=2048, d=4", "rate"))],
    ["Count-Min", "w=16384, d=4", f"상위 100 ARE {cm(16384, 4, 'are_top100'):.2%}, 전체 ARE {cm(16384, 4):.3f}", size(mem("w=16384, d=4")), size(mem("movie dict")), f"{mem('movie dict') / mem('w=16384, d=4'):,.1f}배", rate(tm("w=16384, d=4", "rate"))],
    ["HyperLogLog", "p=14", f"RMSE {hl(14):.2%}", size(mem("p=14")), size(mem("pair set")), f"{mem('pair set') / mem('p=14'):,.0f}배", rate(tm("p=14", "rate"))],
    ["Reservoir (R)", "k=10,000 *", f"TV {rs('R', 10000, 'tv'):.4f}, 평균 오차 {rs('R', 10000, 'mean_err'):.4f}", size(10000 * mem_per), size(mem("전체 저장")), f"{mem('전체 저장') / (10000 * mem_per):,.0f}배", rate(tm("R, k=1000", "rate"))],
]

CSS = """
@page { size: A4; margin: 17mm 16mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif; font-size: 10pt; line-height: 1.62; color: #1b1b1a; margin: 0; word-break: keep-all; background: #fff; }
header { border-bottom: 2px solid #1b1b1a; padding-bottom: 12pt; margin-bottom: 14pt; }
header h1 { font-size: 21pt; line-height: 1.3; margin: 0 0 6pt; letter-spacing: -.4pt; }
.meta { color: #52514e; font-size: 9.5pt; line-height: 1.55; }
h2 { font-size: 14.5pt; margin: 20pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #c9c8c2; break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0 7pt; }
ul { margin: 4pt 0 8pt; padding-left: 15pt; } li { margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; line-height: 1.4; margin: 5pt 0 9pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { white-space: nowrap; text-align: left; font-weight: 700; border-bottom: 1.2px solid #1b1b1a; padding: 3.5pt 5pt; background: #f4f3f0; }
td { border-bottom: .5px solid #d9d8d2; padding: 3pt 5pt; vertical-align: top; }
.num td:not(:first-child), .num th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
.num2 td:nth-child(n+3), .num2 th:nth-child(n+3) { text-align: right; font-variant-numeric: tabular-nums; }
.b { font-weight: 700; color: #1a5fb4; }
figure { margin: 7pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: #52514e; margin-top: 2pt; text-align: left; }
.cap { font-size: 8.6pt; color: #52514e; margin: -5pt 0 9pt; }
.summary { background: #f4f3f0; padding: 9pt 12pt; margin: 0 0 8pt; break-inside: avoid; }
.summary h2 { border: 0; margin: 0 0 3pt; padding: 0; font-size: 11.5pt; }
.summary ul { margin-bottom: 0; }
code, pre { font-family: Menlo, monospace; font-size: 8.2pt; }
pre { background: #f4f3f0; padding: 7pt 9pt; line-height: 1.45; white-space: pre-wrap; margin: 4pt 0 8pt; break-inside: avoid; }
.qa { border-left: 3.5px solid #2a78d6; background: #f2f6fc; padding: 8pt 11pt; margin: 8pt 0 10pt; break-inside: avoid; }
.qa .q { font-weight: 700; font-size: 11pt; margin-bottom: 3pt; color: #123f7a; break-after: avoid; }
.qa .a { font-weight: 700; margin-bottom: 4pt; }
.qa p:last-child, .qa ul:last-child { margin-bottom: 0; }
"""

BODY = f"""
<header>
<h1>스트리밍 알고리즘 4종 구현과<br>정확도·메모리·시간 트레이드오프 분석</h1>
<div class="meta">윤승희 · 학번 2243175 · 2026년 11월 6일<br>소스코드: {GITHUB}</div>
</header>

<div class="summary"><h2>핵심 결과</h2><ul>
<li><b>메모리 절감:</b> HyperLogLog는 {size(mem('p=14'))}로 고유 쌍 {GT['pairs']:,}개를 RMSE {hl(14):.2%}로 추정했습니다. 정확한 Python set({size(mem('pair set'))})의 약 {mem('pair set') / mem('p=14'):,.0f}분의 1입니다.</li>
<li><b>파라미터는 클수록 좋지 않습니다:</b> Bloom Filter는 m/n=4에서 k를 3에서 14로 늘리면 FPR이 {bf(4, 3):.1%}에서 {bf(4, 14):.1%}로 나빠졌고, Count-Min은 같은 메모리에서 depth에 최적점이 있었습니다.</li>
<li><b>시간은 오히려 손해입니다:</b> 순수 Python 구현에서는 정확한 자료구조가 스케치보다 빨랐습니다. 이번 실험에서 스케치의 이점은 메모리입니다.</li>
<li><b>결론:</b> 가장 실용적인 알고리즘은 HyperLogLog입니다. 실제 서비스 로그라면 고유 사용자 집계에 HLL을 먼저 적용하고, 인기 항목 집계에 Count-Min을 더하겠습니다.</li>
</ul></div>

<h2>1. 데이터셋 설명</h2>
<p><b>MovieLens 1M</b>(GroupLens)을 사용했습니다. <code>사용자::영화::평점::시각</code> 형식의 평점 이벤트 {N:,}건으로, 과제 조건인 100,000건 이상의 텍스트 로그를 충족합니다. 파일은 압축을 풀지 않고 한 줄씩 읽었으며, Python 힙 최고 사용량은 <b>{size(ENV['stream_peak'])}</b>로 전체를 적재할 때의 {size(ENV['load_peak'])}보다 훨씬 작습니다. 라이선스가 재배포를 금지하므로 데이터는 저장소에 넣지 않고, 실행 시 내려받아 SHA-256으로 확인합니다.</p>
{table(["항목", "값", "항목", "값"], [
    ["레코드 수", f"{N:,}", "평균 평점", f"{GT['mean_rating']:.3f}"],
    ["고유 사용자", f"{GT['users']:,}", "사용자당 평점 수", f"최소 {GT['min_user_count']} / 중앙값 {GT['median_user_count']:.0f} / 최대 {GT['max_user_count']:,}"],
    ["고유 영화", f"{GT['movies']:,}", "영화당 평점 수", f"최소 {GT['min_movie_count']} / 중앙값 {GT['median_movie_count']:.1f} / 최대 {GT['max_movie_count']:,}"],
    ["고유 (사용자, 영화) 쌍", f"{GT['pairs']:,}", "상위 10% 영화의 평점 비중", f"{GT['top10pct_share']:.1%}"],
])}
{fig("eda", "그림 1. 영화 빈도는 긴 꼬리 분포입니다. 평점 수 10 이하인 영화가 " + f"{GT['movies_le_10']}개입니다. 구간별 평균은 사용자 구성에 따라 3.52~3.64 범위에서 변합니다.")}
<p><b>실험에 영향을 주는 특성이 두 가지 있습니다.</b> 첫째, 파일이 <b>사용자 ID 순으로 정렬</b>되어 있어 앞의 1,000건에는 사용자가 {GT['first_1000_users']}명뿐입니다. 스트림 앞부분만 쓰면 표본이 치우치므로 Reservoir Sampling의 효과를 검증하기 좋습니다. 둘째, 시각 순서가 아니므로(인접 레코드 시각 역전 {GT['timestamp_inversions']:,}회) 실시간 로그의 도착 순서를 그대로 재현하지는 않습니다.</p>

<h2>2. 선택한 알고리즘 개요</h2>
<p>과제는 2개 이상을 요구하지만, 스트리밍 문제 유형별 비교를 위해 4종을 모두 구현했습니다.</p>
{table(["알고리즘", "질문", "자료구조", "오차의 방향과 이론값", "스트림 키"], [
    ["Bloom Filter", "이 원소를 본 적이 있는가?", "m비트 배열, 해시 k개", "거짓 음성 없음. FPR ≈ (1−e<sup>−kn/m</sup>)<sup>k</sup>, 최적 k ≈ 0.693·m/n", "사용자:영화 쌍"],
    ["Count-Min Sketch", "이 항목이 몇 번 나왔는가?", "d×w 카운터", "과대추정만 발생. 확률 1−e<sup>−d</sup> 이상으로 오차 ≤ (e/w)·N", "영화 ID"],
    ["HyperLogLog", "서로 다른 원소가 몇 개인가?", "2<sup>p</sup>개 레지스터", "상대 표준오차 ≈ 1.04/√(2<sup>p</sup>)", "쌍, 사용자, 영화"],
    ["Reservoir Sampling", "균등 표본은 무엇인가?", "크기 k 배열", "모든 원소의 포함 확률이 k/n. 추정 오차는 약 1/√k로 감소", "레코드"],
])}

<h2>3. 구현 방식</h2>
<ul>
<li><b>직접 구현:</b> 네 알고리즘의 핵심 로직은 라이브러리 없이 표준 라이브러리만으로 <code>sketches.py</code>에 구현했습니다. numpy와 scipy는 결과 평가에만 씁니다.</li>
<li><b>해시:</b> 모두 <code>blake2b</code>에 시드를 salt로 넣어 사용합니다. Python 내장 <code>hash()</code>는 실행마다 값이 달라져 재현성을 해치기 때문입니다.</li>
<li><b>Bloom Filter:</b> 이중 해싱(g<sub>i</sub> = h<sub>1</sub> + i·h<sub>2</sub>)으로 한 번의 해시 계산에서 k개 위치를 만듭니다. 조회는 0인 비트를 만나면 바로 끝납니다.</li>
<li><b>Count-Min:</b> 이중 해싱을 쓰면 두 원소가 전 행에서 함께 충돌할 확률이 커지므로, 해시 출력을 행 수만큼 나눠 행별로 독립적인 값을 씁니다. 확장으로 보수적 갱신(Conservative Update)도 구현했습니다.</li>
<li><b>HyperLogLog:</b> 상위 p비트로 레지스터를 고르고 나머지 비트의 첫 1 위치를 기록합니다. 소규모 구간은 선형 계수로 보정합니다. 레지스터는 이론상 6비트면 충분하지만 바이트 단위로 저장했습니다.</li>
<li><b>Reservoir:</b> 원소마다 난수를 뽑는 Algorithm R과, 다음 교체 위치까지 건너뛰는 확장 기법 Algorithm L을 구현했습니다.</li>
<li><b>Ground Truth:</b> 같은 스트림을 한 번 읽어 <code>set</code>과 <code>Counter</code>로 정확한 값을 계산했습니다. Bloom Filter의 음성 질의는 실제 사용자×영화 조합 중 기록이 없는 쌍 1,000,000개를 시드별로 뽑았습니다.</li>
</ul>
<pre># HyperLogLog: 레지스터 선택과 첫 1비트 위치 기록
def add(self, item):
    h = int.from_bytes(blake2b(item.encode(), digest_size=8, salt=self.salt).digest(), "little")
    i = h &gt;&gt; self.q                                # 상위 p비트 = 레지스터 번호
    rank = self.q - (h &amp; self.mask).bit_length() + 1  # 나머지 비트의 첫 1 위치
    if rank &gt; self.reg[i]:
        self.reg[i] = rank

# Count-Min: 갱신은 행마다 +1, 조회는 행들의 최솟값
def query(self, item):
    return min(self.table[c] for c in self._cells(item))</pre>
<p>정확도 실험에서는 알고리즘의 구조적 성질을 함께 확인했습니다. Bloom Filter 전 시행의 거짓 음성은 {int(bloom.fn.sum())}건이고, Count-Min은 모든 영화에서 과소추정이 없었으며(코드의 assert로 검사), Reservoir는 표본 크기와 중복 없음을 매번 검사합니다.</p>

<h2>4. 실험 환경</h2>
{table(["항목", "내용"], [
    ["하드웨어", "Apple M1, 8코어, RAM 8 GB"],
    ["소프트웨어", f"macOS 26.6, CPython {ENV['python']}, numpy {ENV['numpy']}, pandas {ENV['pandas']}"],
    ["정확도 측정", f"시드별 반복(Bloom {ENV['seeds']['bloom']}, Count-Min {ENV['seeds']['cms']}, HLL {ENV['seeds']['hll']}, Reservoir {ENV['seeds']['reservoir']}회). 6개 프로세스 병렬 실행. 재실행 시 결과 파일이 동일함을 확인"],
    ["시간 측정", f"단일 프로세스, 파일에서 직접 스트리밍. 전체 설정을 매 반복 섞어 {ENV['repeats']}회 측정한 중앙값"],
    ["메모리 측정", "tracemalloc의 Python 힙 사용량. 처리 시간을 늘리므로 시간 측정과 별도로 1회 실행"],
])}

<h2>5. 파라미터 설정</h2>
{table(["알고리즘", "비교 파라미터", "시행 수"], [
    ["Bloom Filter", "m/n ∈ {2, 4, 6, 8, 10, 12, 16, 20}, k ∈ {1, 2, 3, 4, 5, 7, 10, 14}", "64조합 × 3 시드"],
    ["Count-Min", "w ∈ {64, 128, …, 32768}, d ∈ {1, 2, 4, 8, 16}. 보수적 갱신은 d ∈ {2, 4}", "70조합 × 5 시드"],
    ["HyperLogLog", "p ∈ {4, …, 16} (레지스터 16~65,536개), 대상 3종", "39조합 × 15 시드"],
    ["Reservoir", "k ∈ {100, 500, 1K, 5K, 10K, 50K, 100K}. R, L, 전체 데이터 무작위 표본(기준), 앞에서 k개(대조군)", "시드 30개, 대조군은 1회"],
])}

<h2>6. 정확도 비교 결과</h2>
<h3>6-1. Bloom Filter</h3>
{fig("bloom", "그림 2. 오른쪽 그림처럼 k에 따라 FPR이 U자로 변하며, 메모리가 클수록 최적 k도 커집니다.")}
{table(["m/n", "크기"] + [f"k={k}" for k in KS], bf_rows, "num2", bf_bold)}
<p class="cap">값은 FPR, 파란 숫자는 행별 최솟값입니다.</p>
<p>측정 FPR은 이론값과 잘 맞습니다. 거짓 양성이 300건 이상인 조합에서 측정값/이론값은 {ratio_lo:.3f}~{ratio_hi:.3f}입니다. m/n=10의 최적 k는 이론값 {10 * .693:.1f}에 가까운 k=7(FPR {bf(10, 7):.3%})입니다. 메모리를 2배로 늘리면(m/n=10→20) FPR은 약 {bf(10, 7) / bf(20, 14):.0f}분의 1이 되어, 메모리 대비 오차 감소가 네 알고리즘 중 가장 빠릅니다.</p>

<h3>6-2. Count-Min Sketch</h3>
{fig("cms", "그림 3. 왼쪽: 메모리와 오차. 가운데: 같은 메모리에서 depth를 바꾼 결과. 오른쪽: 인기 영화와 비인기 영화의 오차 차이.")}
{table(["w"] + [f"d={d}" for d in (1, 2, 4, 8, 16)] + ["d=4 크기"], cm_rows, "num")}
<p class="cap">값은 전체 영화에 대한 평균 상대오차(ARE)입니다.</p>
<ul>
<li><b>오차는 비인기 항목에 몰립니다.</b> w=2048, d=4에서 상위 100 영화의 ARE는 {cm(2048, 4, 'are_top100'):.1%}지만 하위 50%는 {cm(2048, 4, 'are_tail'):.2f}, 즉 실제 값의 약 {cm(2048, 4, 'are_tail') + 1:.0f}배로 추정합니다. 충돌로 더해지는 절대 오차가 항목 크기와 무관하기 때문입니다. Count-Min은 인기 항목 탐지에는 적합하지만 드문 항목의 빈도 추정에는 부적합합니다.</li>
<li><b>이론 보장은 지켜졌습니다.</b> 모든 (w, d)에서 오차가 (e/w)·N을 넘는 영화의 비율이 δ = e<sup>−d</sup> 이하였습니다. 다만 이 경계는 실측 오차보다 느슨합니다.</li>
<li><b>보수적 갱신:</b> w=2048, d=4에서 MAE가 {cm(2048, 4, 'mae'):.1f}→{cm(2048, 4, 'mae', True):.1f}, ARE가 {cm(2048, 4):.2f}→{cm(2048, 4, 'are', True):.2f}로 줄었고, 메모리는 같지만 갱신 시간이 {tm('w=2048, d=4'):.2f}→{tm('w=2048, d=4, CU'):.2f} μs로 늘었습니다.</li>
</ul>

<h3>6-3. HyperLogLog</h3>
{fig("hll", "그림 4. 측정 오차가 이론 곡선을 따릅니다. p가 작으면 시드에 따른 편차가 큽니다.")}
{table(["p", "레지스터", "크기", "이론 오차", "쌍", "사용자", "영화"], hl_rows, "num")}
<p class="cap">값은 상대오차 RMSE이며 시드 15개 기준입니다.</p>
<p>로그-로그 기울기는 {fit_hll:.2f}로 이론값 −0.5와 비슷합니다. 메모리를 4배로 늘려야 오차가 절반이 됩니다. 쌍, 사용자, 영화의 실제 값이 최대 약 270배 차이 나도 오차 수준이 비슷해, 메모리가 카디널리티와 거의 무관하다는 점도 확인했습니다. p=4의 쌍 스트림에서는 평균 편향이 {hl(4, 'pairs', 'bias'):+.1%}로 커서, 이 정도 크기는 실사용에 부적합합니다.</p>

<h3>6-4. Reservoir Sampling</h3>
{fig("reservoir", "그림 5. 왼쪽: 평점 분포 거리. 가운데: 표본이 스트림 전체에 균등하게 분포합니다. 오른쪽: 앞에서 k개를 고르면 사용자 다양성이 크게 떨어집니다.", 92)}
{table(["k", "TV: R", "TV: L", "TV: 전체 무작위", "TV: 앞 k개", "평균 오차: R", "평균 오차: 앞 k개", "사용자: R", "사용자: 앞 k개"], rs_rows, "num")}
<p class="cap">TV는 표본과 전체 평점 분포 사이의 총변동거리이고, 평균 오차는 평균 평점의 절대 오차입니다.</p>
<p>R과 L은 Ground Truth에 해당하는 전체 데이터 무작위 표본과 같은 수준의 오차를 보입니다. 위치 균등성도 검정했습니다. 시행별 KS 검정에서 p &lt; .05 비율은 {ks_rate:.1%}로 우연 기대치 5%와 비슷하고, k=10,000 표본을 합친 20구간 카이제곱 검정의 p값은 R {chi['R']:.2f}, L {chi['L']:.2f}입니다. 균등하지 않다는 증거는 없습니다. 반면 앞에서 k개를 고르면 k=1,000일 때 사용자가 {rs('first', 1000, 'users_in_sample'):.0f}명뿐입니다. 이 대조군은 시드가 없어 1회 값이며, k=500의 낮은 평균 오차({rs('first', 500, 'mean_err'):.4f})는 우연히 맞은 경우입니다. 로그-로그 기울기는 {fit_res:.2f}로 이론값 −0.5에 가깝습니다.</p>

<h2>7. 메모리 사용량 비교 결과</h2>
{fig("mem_time", "그림 6. 대표 설정의 메모리(로그 축)와 처리량. 회색은 정확한 자료구조입니다.")}
{table(["구분", "설정", "처리 후 유지", "최고(스트림 제외)", "자료구조 크기"], m_rows, "num2")}
<p class="cap">유지와 최고 메모리는 tracemalloc 값입니다. 자료구조 크기는 비트 배열·카운터·레지스터의 바이트 수이며, 표본이나 Python 객체는 해당하지 않아 “-”로 표시했습니다.</p>
<p><b>정확한 자료구조와의 비교는 데이터 특성에 좌우됩니다.</b> 측정값은 Python 객체 오버헤드를 포함하므로 set의 {size(mem('pair set'))}은 다소 불리한 비교입니다. 사용자×영화 조합이 유한한 이 데이터에서는 조합마다 1비트씩 쓰는 정확한 비트맵이 약 {size(GT['exact_bitmap_bytes'])}이어서(계산값), m/n=20의 Bloom Filter({size(mem('m/n=20, k=14'))})와 큰 차이가 없습니다. 영화 빈도도 키가 {GT['movies']:,}개뿐이라 정확한 dict가 {size(mem('movie dict'))}로 작고, 오차가 작은 w=16384, d=4({size(mem('w=16384, d=4'))})와 비슷합니다. 즉 키 공간이 작으면 스케치의 이점이 줄고, URL이나 IP처럼 키 공간이 크거나 미리 알 수 없을 때 이점이 커집니다.</p>

<h2>8. 처리 시간 비교 결과</h2>
{table(["구분", "설정", "전체(초)", "최소~최대", "처리량", "순처리(μs/건)"], t_rows, "num2")}
<p class="cap">전체 시간과 처리량은 파일 읽기와 파싱을 포함합니다. 순처리는 파싱만 수행한 기준 시간(쌍 {TM[TM.stream == 'pair_stream'].base_s.iloc[0]:.2f}초, 영화 {TM[TM.stream == 'movie_stream'].base_s.iloc[0]:.2f}초)을 뺀 값입니다.</p>
{table(["조회 작업", "초당 처리량"], q_rows, "num")}
<ul>
<li><b>순수 Python에서는 정확한 구조가 더 빠릅니다.</b> set은 건당 {tm('pair set'):.2f} μs인데 Bloom Filter(k=7)는 {tm('m/n=10, k=7'):.2f} μs입니다. 스케치는 해시를 Python에서 호출하고 위치마다 반복하지만, set은 C로 구현되어 있기 때문입니다. C나 Rust 구현이라면 격차가 달라질 수 있으나 이번에는 측정하지 않았습니다.</li>
<li><b>시간은 메모리 크기가 아니라 해시·갱신 횟수에 좌우됩니다.</b> Count-Min은 d=1→16에서 {tm('w=2048, d=1'):.2f}→{tm('w=2048, d=16'):.2f} μs로 늘었지만, w=2048→16384에서는 {tm('w=2048, d=4'):.2f}→{tm('w=16384, d=4'):.2f} μs로 거의 그대로입니다. Bloom Filter도 k=1→14에서 {tm('m/n=10, k=1'):.2f}→{tm('m/n=10, k=14'):.2f} μs로 늘었습니다. HLL은 p에 관계없이 약 {TM[TM.algo == 'HyperLogLog'].us_per_item.mean():.1f} μs입니다.</li>
<li><b>Bloom Filter 조회:</b> k=14, m/n=20에서 음성 조회({rate(Q['Bloom m/n=20, k=14, 음성'])})가 양성 조회({rate(Q['Bloom m/n=20, k=14, 양성'])})보다 빠릅니다. 0인 비트를 만나면 바로 끝나기 때문입니다.</li>
<li><b>Algorithm L:</b> k=1,000에서 R보다 빠르지만({tm('R, k=1000'):.2f}→{tm('L, k=1000'):.2f} μs), k=100,000에서는 교체가 잦아 차이가 줄어듭니다({tm('R, k=100000'):.2f}→{tm('L, k=100000'):.2f} μs). 파일을 어차피 한 줄씩 읽어야 하므로 전체 처리량 향상은 더 작습니다.</li>
<li><b>측정 잡음:</b> 일부 설정은 최소~최대 폭이 넓습니다. m/n=10, k=14보다 m/n=20, k=14가 빠르게 나온 것처럼 수십 % 이내의 차이는 잡음일 수 있으니 큰 경향만 해석해야 합니다.</li>
</ul>

<h2>9. 알고리즘별 장단점 분석</h2>
{table(["알고리즘", "설정", "정확도", "메모리", "정확한 방식", "절감", "처리량"], summary)}
<p class="cap">* Reservoir의 메모리는 k=100,000 측정값으로 환산했고 처리량은 k=1,000 측정값입니다. Count-Min은 키 공간이 작아 절감 폭이 작습니다.</p>
{table(["알고리즘", "장점", "단점", "적합한 곳"], [
    ["Bloom Filter", "거짓 음성이 없음. 메모리 대비 오차 감소가 가장 빠름", "원소를 꺼내거나 삭제할 수 없음. n을 미리 알아야 m, k를 정할 수 있고 n을 넘기면 FPR 급증", "중복 제거, 캐시 앞단 조회 차단"],
    ["Count-Min", "과대추정만 발생하며 인기 항목에 정확함. 갱신 시간이 w와 무관", "비인기 항목의 상대오차가 큼. 키 공간이 작으면 dict와 차이가 작음. 키 목록은 저장하지 않음", "인기 항목 집계, 트래픽 모니터링"],
    ["HyperLogLog", f"수십 KB로 약 1% 오차. 파라미터가 p 하나이고 오차 예측이 쉬움", "고유 수만 알 수 있고 어떤 원소인지 모름. 소규모 p에서 편향", "순방문자, 고유 IP 집계"],
    ["Reservoir", "원본 레코드가 남아 사후 분석이 자유로움. 구현이 간단하고 정렬된 스트림에도 편향 없음", "메모리가 k에 비례하고 레코드 크기에도 좌우됨. 희귀 항목은 표본에 잘 잡히지 않음", "로그 표본 저장, 디버깅, 분포 점검"],
])}

<h2>10. 결론: 최종 분석 질문</h2>
{fig("tradeoff", f"그림 7. 알고리즘별 정확도-메모리 곡선. 오차 지표가 달라 높이는 비교할 수 없고 기울기만 비교합니다.", 72)}

<div class="qa"><div class="q">Q1. 정확도와 메모리 사이에는 어떤 trade-off가 있었는가?</div>
<div class="a">메모리를 늘리면 오차는 줄지만, 줄어드는 속도는 알고리즘마다 크게 달랐습니다.</div>
<ul>
<li><b>Bloom Filter는 지수적입니다.</b> 최적 k 기준으로 메모리 2배(m/n 10→20)에 FPR이 {bf(10, 7):.3%}→{bf(20, 14):.4%}로 약 {bf(10, 7) / bf(20, 14):.0f}분의 1이 됩니다.</li>
<li><b>HyperLogLog와 Reservoir는 제곱근입니다.</b> 측정 기울기 {fit_hll:.2f}, {fit_res:.2f}로 메모리를 4배 써야 오차가 절반입니다. 그래서 초반 절감은 크지만 높은 정밀도는 비쌉니다.</li>
<li><b>Count-Min은 충돌이 사라지는 구간에서 급격히 좋아집니다.</b> d=1에서는 w에 거의 반비례(기울기 {fit_cms:.2f})하지만, d=4는 w가 키 수 {GT['movies']:,} 근처를 넘으면서 ARE가 {cm(1024, 4):.1f}→{cm(8192, 4):.3f}로 빠르게 떨어집니다.</li>
<li><b>이점의 크기는 데이터에 달려 있습니다.</b> HLL은 정확한 set의 약 {mem('pair set') / mem('p=14'):,.0f}분의 1이지만, Count-Min w=16384, d=4는 dict의 약 {mem('movie dict') / mem('w=16384, d=4'):.1f}분의 1에 그칩니다. 스케치는 키 공간이 클수록 유리합니다.</li>
</ul></div>

<div class="qa"><div class="q">Q2. 파라미터 증가가 항상 성능 향상으로 이어졌는가?</div>
<div class="a">아닙니다. 파라미터를 키워도 정확도가 나빠지거나, 정확도 이득 없이 시간과 메모리만 늘어나는 경우가 있었습니다.</div>
<ul>
<li><b>Bloom Filter의 k: 정확도가 나빠졌습니다.</b> m/n=4에서 k=3의 FPR은 {bf(4, 3):.1%}이지만 k=14는 {bf(4, 14):.1%}입니다. 비트가 채워져 판별력이 사라지기 때문입니다. 처리 시간도 k에 비례해 늘어납니다.</li>
<li><b>Count-Min의 d: 메모리가 고정되면 최적점이 있습니다.</b> 카운터 16,384개에서 d=1, 2, 4, 8, 16의 ARE는 {", ".join(f"{x:.2f}" for x in d14)}입니다. d를 너무 키우면 w가 줄어 오차가 다시 커지고, 시간은 계속 늘어납니다.</li>
<li><b>메모리만 늘리는 경우엔 수렴합니다.</b> d=4에서 w=16384→32768은 ARE가 {cm(16384, 4):.3f}→{cm(32768, 4):.3f}로 이미 작은 오차를 조금 더 줄일 뿐, 메모리는 2배입니다.</li>
<li><b>HLL의 p, Reservoir의 k:</b> 정확도는 대체로 개선되지만 제곱근 법칙으로 이득이 줄어듭니다. HLL은 p=14→16에서 메모리가 4배인데 RMSE는 {hl(14):.2%}→{hl(16):.2%}로 줄어드는 데 그칩니다.</li>
</ul></div>

<div class="qa"><div class="q">Q3. 어떤 알고리즘이 가장 실용적이라고 판단되는가?</div>
<div class="a">HyperLogLog입니다. 네 가지 중 메모리 대비 효과가 가장 크고 설정이 가장 쉽습니다.</div>
<p>{size(mem('p=14'))}로 {GT['pairs']:,}개의 고유 쌍을 RMSE {hl(14):.2%}로 추정했고, 대상 크기가 최대 약 270배 달라도 오차가 비슷했습니다. 파라미터는 p 하나이며 1.04/√m으로 필요한 크기를 미리 계산할 수 있습니다. Bloom Filter는 n을 미리 알아야 하고, Count-Min은 드문 항목의 오차가 크며 이번 데이터에서는 dict 대비 이점도 작았습니다. 다만 이 판단은 “고유 개수”라는 질문에 한정됩니다. 포함 여부나 빈도를 물어야 한다면 HLL로는 답할 수 없습니다.</p></div>

<div class="qa"><div class="q">Q4. 실제 서비스 로그 분석에 적용한다면 어떤 알고리즘을 선택할 것인가?</div>
<div class="a">HyperLogLog를 먼저 도입하고, 인기 항목 집계에 Count-Min Sketch를 더하겠습니다.</div>
<ul>
<li><b>1순위 HyperLogLog (DAU, 페이지별 순방문자, 고유 IP):</b> 서비스 로그에서 가장 흔하고 정확한 집계 비용이 가장 큰 지표입니다. p=14 정도면 지표당 약 16 KB로 약 1% 오차를 기대할 수 있어, 지표가 많아도 부담이 작습니다.</li>
<li><b>2순위 Count-Min (인기 URL, 트래픽 급증 키):</b> 키 공간이 크고 관심 대상이 상위 항목이라 이번에 확인한 “상위 항목에 정확함”이 그대로 강점이 됩니다. 단, 스케치에는 키 목록이 없어 상위 항목을 뽑으려면 별도의 힙이 필요합니다.</li>
<li><b>보조 Reservoir (장애 조사용 표본):</b> 정렬되거나 치우친 로그에서도 편향 없는 원본 표본을 남길 수 있습니다.</li>
<li><b>필요할 때 Bloom Filter (이벤트 중복 제거):</b> 거짓 양성은 신규 이벤트를 중복으로 잘못 버리는 결과가 되므로, 데이터 손실을 허용할 수 있을 때만 쓰겠습니다.</li>
</ul>
<p>운영에서는 Python 구현을 그대로 쓰지 않겠습니다. 이번 측정에서 처리량이 약 {tm('p=14', 'rate') / 1e3:,.0f}K건/초로, 단일 프로세스 Python에서는 대규모 스트림을 감당하기 어렵습니다. Redis의 HLL처럼 검증된 네이티브 구현을 쓰는 편이 낫습니다. 시간별 HLL을 합치는 병합 연산도 실무에서 중요하지만 이번 과제에서는 구현하거나 측정하지 않았습니다.</p></div>

<h3>한계</h3>
<ul>
<li>MovieLens 1M은 키 공간이 작고 쌍이 모두 고유합니다(같은 쌍의 반복이 없음). 그래서 Count-Min의 메모리 이점을 작게, Python set과의 비교에서 Bloom Filter·HLL의 이점을 크게 보여 줄 수 있습니다.</li>
<li>파일이 시간순이 아니어서 실제 로그의 시간적 변화나 트래픽 급증은 재현하지 못했습니다.</li>
<li>Bloom Filter의 FPR이 매우 낮은 조합은 거짓 양성이 수십~수백 건뿐이라 상대적으로 불확실합니다.</li>
<li>시간은 M1 한 대의 순수 Python 값이므로, 절대값보다 설정 간 경향을 봐야 합니다.</li>
</ul>

<h2>재현 방법</h2>
<p>소스코드: {GITHUB}</p>
<pre>python3.12 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt
.venv/bin/python sketches.py          # 알고리즘 자체 검증
.venv/bin/python run_experiments.py   # 데이터 다운로드와 전체 실험 (M1 기준 약 20분)
.venv/bin/python build_report.py &lt;github-url&gt;   # 그림과 이 PDF 생성</pre>
"""

doc = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>스트리밍 알고리즘 분석</title><style>{CSS}</style></head><body>{BODY}</body></html>'
OUT.parent.mkdir(exist_ok=True)
tmp = OUT.with_suffix(".html")
tmp.write_text(doc)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox", f"--print-to-pdf={OUT}", tmp.as_uri()],
               check=True, capture_output=True, timeout=120)
tmp.unlink()
print(OUT)
