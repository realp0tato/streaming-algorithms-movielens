"""스트리밍 알고리즘 4종 직접 구현. 표준 라이브러리만 사용한다. `python sketches.py`로 자체 검증.

해시는 모두 blake2b(시드는 salt)로 통일했다. 속도보다 알고리즘 간 공정한 비교와 재현성을 우선한 선택이다.
"""
import math
import random
import struct
from array import array
from hashlib import blake2b


def _salt(seed):
    return seed.to_bytes(16, "little")


class BloomFilter:
    """m비트 배열과 k개 해시로 집합 포함 여부를 근사한다. 거짓 음성은 없고 거짓 양성만 있다."""

    def __init__(self, m, k, seed=0):
        self.m, self.k, self.salt = m, k, _salt(seed)
        self.bits = bytearray((m + 7) // 8)

    def _base(self, item):
        # Kirsch-Mitzenmacher 이중 해싱: g_i = h1 + i*h2. 해시 계산 1회로 k개 위치를 만든다.
        h1, h2 = struct.unpack("<QQ", blake2b(item.encode(), digest_size=16, salt=self.salt).digest())
        return h1, h2 | 1

    def add(self, item):
        h1, h2 = self._base(item)
        m, bits = self.m, self.bits
        for i in range(self.k):
            x = (h1 + i * h2) % m
            bits[x >> 3] |= 1 << (x & 7)

    def __contains__(self, item):
        h1, h2 = self._base(item)
        m, bits = self.m, self.bits
        for i in range(self.k):
            x = (h1 + i * h2) % m
            if not bits[x >> 3] >> (x & 7) & 1:
                return False
        return True

    def fill_ratio(self):
        return int.from_bytes(self.bits, "little").bit_count() / self.m

    @property
    def nbytes(self):
        return len(self.bits)


class CountMinSketch:
    """d행 w열 카운터로 빈도를 근사한다. 추정값은 실제값 이상이다(과대추정만 발생).

    conservative=True는 Estan-Varghese의 보수적 갱신: 현재 최솟값+1 까지만 올려 충돌 오차를 줄인다.
    이중 해싱은 w^2 확률로 전 행이 함께 충돌하므로, 행마다 digest의 독립 구간을 쓴다.
    """

    def __init__(self, w, d, seed=0, conservative=False):
        assert 1 <= d <= 16, "blake2b 64바이트에서 행당 32비트를 쓰므로 d는 16까지"
        self.w, self.d, self.salt, self.conservative = w, d, _salt(seed), conservative
        self.table = array("I", [0]) * (w * d)
        self.fmt = f"<{d}I"
        self.N = 0

    def _cells(self, item):
        hs = struct.unpack(self.fmt, blake2b(item.encode(), digest_size=4 * self.d, salt=self.salt).digest())
        w = self.w
        return [r * w + h % w for r, h in enumerate(hs)]

    def add(self, item):
        t = self.table
        cells = self._cells(item)
        self.N += 1
        if self.conservative:
            target = min(t[c] for c in cells) + 1
            for c in cells:
                if t[c] < target:
                    t[c] = target
        else:
            for c in cells:
                t[c] += 1

    def query(self, item):
        t = self.table
        return min(t[c] for c in self._cells(item))

    @property
    def nbytes(self):
        return len(self.table) * self.table.itemsize


class HyperLogLog:
    """2^p개 레지스터에 '첫 1비트 위치'의 최댓값을 기록해 고유 원소 수를 추정한다."""

    def __init__(self, p, seed=0):
        assert 4 <= p <= 18
        self.p, self.m, self.salt = p, 1 << p, _salt(seed)
        self.reg = bytearray(self.m)  # 이론상 6비트면 충분하나 구현 단순성을 위해 바이트 단위로 저장
        self.q = 64 - p
        self.mask = (1 << self.q) - 1

    def add(self, item):
        h = int.from_bytes(blake2b(item.encode(), digest_size=8, salt=self.salt).digest(), "little")
        i = h >> self.q
        rank = self.q - (h & self.mask).bit_length() + 1
        if rank > self.reg[i]:
            self.reg[i] = rank

    def estimate(self):
        m = self.m
        alpha = {16: .673, 32: .697, 64: .709}.get(m, .7213 / (1 + 1.079 / m))
        raw = alpha * m * m / sum(2.0 ** -r for r in self.reg)
        zeros = self.reg.count(0)
        if raw <= 2.5 * m and zeros:  # 소규모 구간은 선형 계수로 보정
            return m * math.log(m / zeros)
        return raw  # 64비트 해시라 대규모 구간 보정은 불필요

    @property
    def nbytes(self):
        return len(self.reg)


class ReservoirR:
    """Vitter Algorithm R: 원소마다 난수 1개. i번째 원소는 k/i 확률로 표본에 들어간다."""

    def __init__(self, k, seed=0):
        self.k, self.rng, self.n, self.sample = k, random.Random(seed), 0, []

    def add(self, item):
        n = self.n
        self.n = n + 1
        if n < self.k:
            self.sample.append(item)
        else:
            j = self.rng.randrange(n + 1)
            if j < self.k:
                self.sample[j] = item


class ReservoirL:
    """Li의 Algorithm L: 다음 교체 위치까지 건너뛸 개수를 바로 뽑아 난수 사용을 O(k log(n/k))로 줄인다."""

    def __init__(self, k, seed=0):
        self.k, self.rng, self.n, self.sample = k, random.Random(seed), 0, []
        self.w = math.exp(math.log(self.rng.random()) / k)
        self.next = k + self._skip()

    def _skip(self):
        return int(math.log(self.rng.random()) / math.log(1 - self.w))

    def add(self, item):
        n = self.n
        self.n = n + 1
        if n < self.k:
            self.sample.append(item)
        elif n == self.next:
            self.sample[self.rng.randrange(self.k)] = item
            self.w *= math.exp(math.log(self.rng.random()) / self.k)
            self.next = n + 1 + self._skip()


if __name__ == "__main__":
    items = [f"item{i}" for i in range(20000)]

    bf = BloomFilter(10 * len(items), 7, seed=1)
    for x in items:
        bf.add(x)
    assert all(x in bf for x in items), "Bloom 거짓 음성 발생"
    fpr = sum(f"other{i}" in bf for i in range(50000)) / 50000
    assert fpr < .03, fpr  # 이론값 약 0.008

    stream = [f"k{i % 50}" for i in range(5000)] + [f"k{i}" for i in range(1000)]
    truth = {x: stream.count(x) for x in set(stream)}
    for cu in (False, True):
        cms = CountMinSketch(64, 4, seed=1, conservative=cu)
        for x in stream:
            cms.add(x)
        assert all(cms.query(x) >= c for x, c in truth.items()), "CMS 과소추정 발생"
        assert cms.N == len(stream)

    h = HyperLogLog(14, seed=1)
    for x in items:
        h.add(x)
    h.add(items[0])
    assert abs(h.estimate() / len(items) - 1) < .05, h.estimate()
    small = HyperLogLog(14, seed=1)
    for x in items[:100]:
        small.add(x)
    assert abs(small.estimate() - 100) < 5, small.estimate()  # 선형 계수 구간

    # 두 저수지 표본 모두 포함 확률이 k/n 이어야 한다
    for cls in (ReservoirR, ReservoirL):
        hits = [0] * 100
        for seed in range(3000):
            r = cls(10, seed=seed)
            for i in range(100):
                r.add(i)
            assert len(r.sample) == 10 and len(set(r.sample)) == 10
            for i in r.sample:
                hits[i] += 1
        assert all(abs(c / 3000 - .1) < .025 for c in hits), (cls.__name__, min(hits), max(hits))
    print("self-check ok")
