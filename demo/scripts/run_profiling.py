"""SLA Profiler for Dynamo PD deployment.

Profiles prefill and decode engines through the frontend (processor) endpoint
using sglang.bench_serving. Generates profiling data compatible with the RBG
planner's perf_interpolation module.

Output: prefill_raw_data.json + decode_raw_data.json → ConfigMap update.
"""

import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Configuration (from env) ───────────────────────────────────────────────
BASE_URL = os.environ.get(
    "BASE_URL",
    "http://sglang-pd-inference-processor-0.s-sglang-pd-inference-processor.demo.svc.cluster.local:8000",
)
MODEL = os.environ.get("MODEL", "qwen3-6")
TOKENIZER = os.environ.get("TOKENIZER", "/models/qwen-qwen3.6-35b-a3b/main")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
MAX_KV_TOKENS = int(os.environ.get("MAX_KV_TOKENS", "1529490"))
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", "32768"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/profiling_results")
CONFIGMAP_NAME = os.environ.get("CONFIGMAP_NAME", "sglang-pd-inference-profiling")
NAMESPACE = os.environ.get("NAMESPACE", "demo")
UPDATE_CONFIGMAP = os.environ.get("UPDATE_CONFIGMAP", "true").lower() == "true"

# Profiling parameters
PREFILL_ISL_POINTS = [int(x) for x in os.environ.get(
    "PREFILL_ISL_POINTS", "128,256,512,1024,1536,2048,3072,4096"
).split(",")]
PREFILL_OSL = 5  # Very short output to isolate prefill time
PREFILL_NUM_PROMPTS = 5  # Requests per ISL point for averaging

DECODE_OSL = 500  # Fixed OSL for decode measurement
DECODE_ISL_POINTS = [int(x) for x in os.environ.get(
    "DECODE_ISL_POINTS", "256,512,1024,2048,3072"
).split(",")]
DECODE_CONCURRENCY_GRANULARITY = int(os.environ.get("DECODE_GRANULARITY", "6"))


def run_bench(
    num_prompts: int,
    request_rate: float,
    input_len: int,
    output_len: int,
    result_file: str,
) -> dict | None:
    """Run sglang.bench_serving and parse JSON output."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang-oai",
        "--base-url", BASE_URL,
        "--model", MODEL,
        "--tokenizer", TOKENIZER,
        "--dataset-name", "random-ids",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "1.0",  # Fixed length, no variance
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--disable-tqdm",
        "--output-file", result_file,
    ]

    logger.info(f"  bench: prompts={num_prompts} rate={request_rate} isl={input_len} osl={output_len}")

    for attempt in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                with open(result_file, "r") as f:
                    # bench_serving writes JSONL; take last valid line
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if lines:
                        return json.loads(lines[-1])
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"  Failed to parse result file: {e}, attempt {attempt+1}")
        else:
            logger.warning(f"  bench failed (exit {proc.returncode}), attempt {attempt+1}")
            if proc.stderr:
                logger.warning(f"  stderr: {proc.stderr[-500:]}")
            time.sleep(5)

    logger.error(f"  All attempts failed for isl={input_len} osl={output_len}")
    return None


def profile_prefill() -> dict:
    """Profile prefill performance at various ISL points.

    Sends low-concurrency requests with minimal OSL to isolate prefill time.
    Measures TTFT (time to first token) which is dominated by prefill computation.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Prefill Profiling")
    logger.info("=" * 60)

    prefill_isl = []
    prefill_ttft = []  # in milliseconds
    prefill_thpt_per_gpu = []

    for isl in PREFILL_ISL_POINTS:
        if isl >= MAX_CONTEXT_LENGTH - 512:
            logger.info(f"  Skipping ISL={isl} (exceeds max_context_length)")
            continue

        result_file = os.path.join(OUTPUT_DIR, f"prefill_isl{isl}.json")
        result = run_bench(
            num_prompts=PREFILL_NUM_PROMPTS,
            request_rate=float("inf"),  # Burst: send all at once
            input_len=isl,
            output_len=PREFILL_OSL,
            result_file=result_file,
        )

        if result is None:
            continue

        # Extract TTFT (bench_serving reports in seconds → convert to ms)
        avg_ttft_s = result.get("mean_ttft_ms")  # actually in ms already
        if avg_ttft_s is None:
            # Try alternative field names
            avg_ttft_s = result.get("avg_ttft_ms") or result.get("mean_ttft")
        if avg_ttft_s is None:
            logger.warning(f"  No TTFT field found for ISL={isl}, keys: {list(result.keys())}")
            continue

        ttft_ms = float(avg_ttft_s)
        if ttft_ms <= 0:
            logger.warning(f"  Invalid TTFT={ttft_ms} for ISL={isl}")
            continue

        # thpt_per_gpu = tokens / time / gpus
        thpt = isl / ttft_ms * 1000 / NUM_GPUS  # tokens/s/gpu

        prefill_isl.append(isl)
        prefill_ttft.append(ttft_ms)
        prefill_thpt_per_gpu.append(round(thpt, 2))

        logger.info(f"  ISL={isl}: TTFT={ttft_ms:.2f}ms, thpt/gpu={thpt:.1f} tok/s")

    data = {
        "prefill_isl": prefill_isl,
        "prefill_ttft": prefill_ttft,
        "prefill_thpt_per_gpu": prefill_thpt_per_gpu,
    }
    logger.info(f"Prefill profiling done: {len(prefill_isl)} data points")
    return data


def profile_decode() -> dict:
    """Profile decode performance across (kv_usage, context_length) grid.

    For each ISL, sweeps concurrency from low to high, measuring ITL and
    throughput. KV usage = (context_len * concurrency) / max_kv_tokens.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Decode Profiling")
    logger.info("=" * 60)

    x_kv_usage = []
    y_context_length = []
    z_itl = []  # in milliseconds
    z_thpt_per_gpu = []

    for isl in DECODE_ISL_POINTS:
        if isl + DECODE_OSL >= MAX_CONTEXT_LENGTH:
            logger.info(f"  Skipping ISL={isl} (isl+osl exceeds context length)")
            continue

        context_length = isl + DECODE_OSL // 2
        max_concurrency = min(
            MAX_KV_TOKENS // (isl + DECODE_OSL),
            512,  # Respect engine's max_running_requests
        )

        if max_concurrency < 1:
            logger.warning(f"  ISL={isl}: max_concurrency=0, skipping")
            continue

        # Generate concurrency sweep points
        if max_concurrency <= DECODE_CONCURRENCY_GRANULARITY:
            concurrency_points = list(range(1, max_concurrency + 1))
        else:
            step = (max_concurrency - 1) / (DECODE_CONCURRENCY_GRANULARITY - 1)
            concurrency_points = [
                max(1, int(1 + i * step))
                for i in range(DECODE_CONCURRENCY_GRANULARITY)
            ]
            # Deduplicate
            concurrency_points = sorted(set(concurrency_points))

        logger.info(f"  ISL={isl}: context_len={context_length}, max_conc={max_concurrency}, sweep={concurrency_points}")

        for conc in concurrency_points:
            result_file = os.path.join(OUTPUT_DIR, f"decode_isl{isl}_conc{conc}.json")
            result = run_bench(
                num_prompts=conc,
                request_rate=float("inf"),  # All concurrent
                input_len=isl,
                output_len=DECODE_OSL,
                result_file=result_file,
            )

            if result is None:
                continue

            # Extract ITL (ms) and output throughput
            avg_itl_ms = result.get("mean_itl_ms") or result.get("avg_itl_ms") or result.get("mean_tpot_ms")
            output_throughput = result.get("output_throughput")  # tokens/s total

            if avg_itl_ms is None or output_throughput is None:
                logger.warning(
                    f"  No ITL/throughput for ISL={isl} conc={conc}, keys: {list(result.keys())}"
                )
                continue

            itl_ms = float(avg_itl_ms)
            thpt_per_gpu = float(output_throughput) / NUM_GPUS
            kv_usage = (isl + DECODE_OSL / 2) * conc / MAX_KV_TOKENS

            x_kv_usage.append(round(kv_usage, 6))
            y_context_length.append(context_length)
            z_itl.append(round(itl_ms, 3))
            z_thpt_per_gpu.append(round(thpt_per_gpu, 2))

            logger.info(
                f"    conc={conc}: kv_usage={kv_usage:.4f}, ITL={itl_ms:.2f}ms, "
                f"thpt/gpu={thpt_per_gpu:.1f} tok/s"
            )

    data = {
        "x_kv_usage": x_kv_usage,
        "y_context_length": y_context_length,
        "z_itl": z_itl,
        "z_thpt_per_gpu": z_thpt_per_gpu,
        "max_kv_tokens": MAX_KV_TOKENS,
    }
    logger.info(f"Decode profiling done: {len(x_kv_usage)} data points")
    return data


def update_configmap(prefill_data: dict, decode_data: dict):
    """Update Kubernetes ConfigMap with profiling results."""
    logger.info(f"Updating ConfigMap {NAMESPACE}/{CONFIGMAP_NAME}...")

    prefill_json = json.dumps(prefill_data, indent=2)
    decode_json = json.dumps(decode_data, indent=2)

    # Use kubectl to patch the configmap
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        pf.write(prefill_json)
        prefill_file = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as df:
        df.write(decode_json)
        decode_file = df.name

    # Delete and recreate configmap
    cmd_delete = [
        "kubectl", "delete", "configmap", CONFIGMAP_NAME,
        "-n", NAMESPACE, "--ignore-not-found",
    ]
    subprocess.run(cmd_delete, capture_output=True)

    cmd_create = [
        "kubectl", "create", "configmap", CONFIGMAP_NAME,
        "-n", NAMESPACE,
        f"--from-file=prefill_raw_data.json={prefill_file}",
        f"--from-file=decode_raw_data.json={decode_file}",
    ]
    proc = subprocess.run(cmd_create, capture_output=True, text=True)
    if proc.returncode == 0:
        logger.info(f"ConfigMap {CONFIGMAP_NAME} updated successfully")
    else:
        logger.error(f"Failed to update ConfigMap: {proc.stderr}")

    os.unlink(prefill_file)
    os.unlink(decode_file)


def main():
    logger.info("SLA Profiler starting")
    logger.info(f"  BASE_URL: {BASE_URL}")
    logger.info(f"  MODEL: {MODEL}")
    logger.info(f"  NUM_GPUS: {NUM_GPUS}")
    logger.info(f"  MAX_KV_TOKENS: {MAX_KV_TOKENS}")
    logger.info(f"  MAX_CONTEXT_LENGTH: {MAX_CONTEXT_LENGTH}")
    logger.info(f"  Prefill ISL points: {PREFILL_ISL_POINTS}")
    logger.info(f"  Decode ISL points: {DECODE_ISL_POINTS}")
    logger.info(f"  Decode concurrency granularity: {DECODE_CONCURRENCY_GRANULARITY}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: Prefill
    prefill_data = profile_prefill()

    # Phase 2: Decode
    decode_data = profile_decode()

    # Save locally
    prefill_path = os.path.join(OUTPUT_DIR, "prefill_raw_data.json")
    decode_path = os.path.join(OUTPUT_DIR, "decode_raw_data.json")
    with open(prefill_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_path, "w") as f:
        json.dump(decode_data, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_DIR}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Prefill: {len(prefill_data['prefill_isl'])} points")
    for i, isl in enumerate(prefill_data["prefill_isl"]):
        logger.info(
            f"  ISL={isl}: TTFT={prefill_data['prefill_ttft'][i]:.1f}ms, "
            f"thpt/gpu={prefill_data['prefill_thpt_per_gpu'][i]:.0f} tok/s"
        )
    logger.info(f"Decode: {len(decode_data['x_kv_usage'])} points")
    if decode_data["z_thpt_per_gpu"]:
        logger.info(
            f"  thpt/gpu range: {min(decode_data['z_thpt_per_gpu']):.0f} - "
            f"{max(decode_data['z_thpt_per_gpu']):.0f} tok/s"
        )
        logger.info(
            f"  ITL range: {min(decode_data['z_itl']):.1f} - "
            f"{max(decode_data['z_itl']):.1f} ms"
        )

    # Update ConfigMap
    if UPDATE_CONFIGMAP:
        update_configmap(prefill_data, decode_data)
    else:
        logger.info("Skipping ConfigMap update (UPDATE_CONFIGMAP=false)")

    logger.info("Profiling complete!")


if __name__ == "__main__":
    main()
