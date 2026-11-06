# 스트리밍 알고리즘 4종 구현과 트레이드오프 분석

MovieLens 1M 평점 스트림으로 Bloom Filter, Count-Min Sketch, HyperLogLog, Reservoir Sampling의 정확도, 메모리, 처리 시간을 비교한 과제입니다. 결과 해석은 `report/`의 PDF를 보세요.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python sketches.py                     # 알고리즘 자체 검증
.venv/bin/python run_experiments.py              # 데이터 다운로드와 전체 실험 → results/
.venv/bin/python build_report.py <github-url>    # 그림과 PDF 생성 (Google Chrome 필요)
```

- `sketches.py`: 알고리즘 직접 구현(표준 라이브러리만 사용).
- `trials.py`: 파라미터별 정확도 시행.
- `run_experiments.py`: Ground Truth, 정확도, 시간, 메모리 측정.
- 데이터: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). 라이선스상 재배포가 금지되어 저장소에 포함하지 않으며, 실행 시 내려받습니다.
