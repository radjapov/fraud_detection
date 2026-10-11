#!/usr/bin/env python3
"""
Response time of the running service, measured over HTTP like a client would see it.

  predict              POST /api/predict (as shipped: also appends the request to history.json)
  predict_no_history   the same with the history write switched off
  shap_worker          POST /api/shap, a new Python process per request (the published design)
  shap_pool            POST /api/shap, one long-lived worker process (cold start reported apart)
  shap_inprocess       POST /api/shap, explainer inside the web process

Rows are transactions from the test period. Results: runs/latency/latency.json

  python -m experiments.bench_latency --n 200 --n-worker 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
        status = r.status
    ms = (time.perf_counter() - t0) * 1000
    if status != 200:
        raise RuntimeError(f"{url} -> {status}: {body[:200]}")
    return ms


def stats(ms):
    a = np.asarray(ms)
    return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.quantile(a, 0.95)),
                mean_ms=float(a.mean()), min_ms=float(a.min()), max_ms=float(a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-worker", type=int, default=30)
    ap.add_argument("--out", default="runs/latency")
    ap.add_argument("--port", type=int, default=5099)
    a = ap.parse_args()

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    from werkzeug.serving import make_server
    from fraud_app import services
    from fraud_app.app import create_app

    app = create_app()
    srv = make_server("127.0.0.1", a.port, app, threaded=False)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{a.port}/api"

    frame = services._demo_frame()
    rows = frame.drop(columns=["is_fraud"]).sample(n=a.n, random_state=0)
    rows = [{k: services.numpy_to_native(v) for k, v in r.items()} for r in rows.to_dict("records")]
    for r in rows:  # JSON cannot carry NaN; missing values travel as null
        for k, v in r.items():
            if isinstance(v, float) and v != v:
                r[k] = None

    res = {}
    hist_backup = services.HISTORY_FILE.read_bytes() if services.HISTORY_FILE.exists() else None
    try:
        for r in rows[:5]:
            post(f"{base}/predict", r)  # warm-up
        res["predict"] = stats([post(f"{base}/predict", r) for r in rows])
        real_save = services.save_history_entry
        import fraud_app.api as api
        api.save_history_entry = lambda e: None
        res["predict_no_history"] = stats([post(f"{base}/predict", r) for r in rows])
        api.save_history_entry = real_save
    finally:
        if hist_backup is not None:
            services.HISTORY_FILE.write_bytes(hist_backup)

    services.SHAP_MODE = "worker"
    res["shap_worker"] = stats([post(f"{base}/shap", r) for r in rows[: a.n_worker]])

    services.SHAP_MODE = "pool"
    services.SHAP_POOL.close()
    cold = post(f"{base}/shap", rows[0])
    res["shap_pool"] = dict(stats([post(f"{base}/shap", r) for r in rows]), cold_start_ms=cold)
    services.SHAP_POOL.close()

    services.SHAP_MODE = "inprocess"
    for r in rows[:5]:
        post(f"{base}/shap", r)
    res["shap_inprocess"] = stats([post(f"{base}/shap", r) for r in rows])

    srv.shutdown()
    env = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
               cpu_count=os.cpu_count())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(dict(environment=env, results=res), indent=2))
    print(f"{'endpoint':<22}{'median ms':>11}{'p95 ms':>10}{'n':>6}")
    for k, v in res.items():
        print(f"{k:<22}{v['median_ms']:>11.1f}{v['p95_ms']:>10.1f}{v['n']:>6}"
              + (f"   cold start {v['cold_start_ms']:.0f} ms" if "cold_start_ms" in v else ""))


if __name__ == "__main__":
    main()
