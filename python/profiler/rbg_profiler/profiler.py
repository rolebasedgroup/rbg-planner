"""SLA Profiler for PD-disaggregated inference.

Profiles prefill and decode engines through the frontend endpoint using
sglang.bench_serving. Generates profiling data compatible with the RBG
planner's perf_interpolation module.

Output: prefill_raw_data.json + decode_raw_data.json -> ConfigMap update.
"""

import json
import logging
import os
import subprocess
import sys
import time

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

logger = logging.getLogger(__name__)


def run_bench(
    base_url: str,
    model: str,
    tokenizer: str,
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
        "--base-url", base_url,
        "--model", model,
        "--tokenizer", tokenizer,
        "--dataset-name", "random-ids",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "1.0",
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


def profile_prefill(
    base_url: str,
    model: str,
    tokenizer: str,
    num_gpus: int,
    max_context_length: int,
    output_dir: str,
    isl_points: list[int],
    num_prompts: int = 5,
    osl: int = 5,
) -> dict:
    """Profile prefill performance at various ISL points.

    Sends low-concurrency requests with minimal OSL to isolate prefill time.
    Measures TTFT (time to first token) which is dominated by prefill computation.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Prefill Profiling")
    logger.info("=" * 60)

    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []

    for isl in isl_points:
        if isl >= max_context_length - 512:
            logger.info(f"  Skipping ISL={isl} (exceeds max_context_length)")
            continue

        result_file = os.path.join(output_dir, f"prefill_isl{isl}.json")
        result = run_bench(
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            num_prompts=num_prompts,
            request_rate=float("inf"),
            input_len=isl,
            output_len=osl,
            result_file=result_file,
        )

        if result is None:
            continue

        ttft_ms = result.get("mean_ttft_ms")
        if ttft_ms is None:
            ttft_ms = result.get("avg_ttft_ms") or result.get("mean_ttft")
        if ttft_ms is None:
            logger.warning(f"  No TTFT field found for ISL={isl}, keys: {list(result.keys())}")
            continue

        ttft_ms = float(ttft_ms)
        if ttft_ms <= 0:
            logger.warning(f"  Invalid TTFT={ttft_ms} for ISL={isl}")
            continue

        thpt = isl / ttft_ms * 1000 / num_gpus

        prefill_isl.append(isl)
        prefill_ttft.append(round(ttft_ms, 2))
        prefill_thpt_per_gpu.append(round(thpt, 2))

        logger.info(f"  ISL={isl}: TTFT={ttft_ms:.2f}ms, thpt/gpu={thpt:.1f} tok/s")

    data = {
        "prefill_isl": prefill_isl,
        "prefill_ttft": prefill_ttft,
        "prefill_thpt_per_gpu": prefill_thpt_per_gpu,
    }
    logger.info(f"Prefill profiling done: {len(prefill_isl)} data points")
    return data


def profile_decode(
    base_url: str,
    model: str,
    tokenizer: str,
    num_gpus: int,
    max_kv_tokens: int,
    max_context_length: int,
    output_dir: str,
    isl_points: list[int],
    osl: int = 500,
    concurrency_granularity: int = 6,
) -> dict:
    """Profile decode performance across (kv_usage, context_length) grid.

    For each ISL, sweeps concurrency from low to high, measuring ITL and
    throughput. KV usage = (context_len * concurrency) / max_kv_tokens.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Decode Profiling")
    logger.info("=" * 60)

    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []

    for isl in isl_points:
        if isl + osl >= max_context_length:
            logger.info(f"  Skipping ISL={isl} (isl+osl exceeds context length)")
            continue

        context_length = isl + osl // 2
        max_concurrency = min(max_kv_tokens // (isl + osl), 512)

        if max_concurrency < 1:
            logger.warning(f"  ISL={isl}: max_concurrency=0, skipping")
            continue

        if max_concurrency <= concurrency_granularity:
            concurrency_points = list(range(1, max_concurrency + 1))
        else:
            step = (max_concurrency - 1) / (concurrency_granularity - 1)
            concurrency_points = sorted(set(
                max(1, int(1 + i * step))
                for i in range(concurrency_granularity)
            ))

        logger.info(f"  ISL={isl}: context_len={context_length}, max_conc={max_concurrency}, sweep={concurrency_points}")

        for conc in concurrency_points:
            result_file = os.path.join(output_dir, f"decode_isl{isl}_conc{conc}.json")
            result = run_bench(
                base_url=base_url,
                model=model,
                tokenizer=tokenizer,
                num_prompts=conc,
                request_rate=float("inf"),
                input_len=isl,
                output_len=osl,
                result_file=result_file,
            )

            if result is None:
                continue

            avg_itl_ms = result.get("mean_itl_ms") or result.get("avg_itl_ms") or result.get("mean_tpot_ms")
            output_throughput = result.get("output_throughput")

            if avg_itl_ms is None or output_throughput is None:
                logger.warning(f"  No ITL/throughput for ISL={isl} conc={conc}, keys: {list(result.keys())}")
                continue

            itl_ms = float(avg_itl_ms)
            thpt_per_gpu = float(output_throughput) / num_gpus
            kv_usage = (isl + osl / 2) * conc / max_kv_tokens

            x_kv_usage.append(round(kv_usage, 6))
            y_context_length.append(context_length)
            z_itl.append(round(itl_ms, 3))
            z_thpt_per_gpu.append(round(thpt_per_gpu, 2))

            logger.info(f"    conc={conc}: kv_usage={kv_usage:.4f}, ITL={itl_ms:.2f}ms, thpt/gpu={thpt_per_gpu:.1f} tok/s")

    data = {
        "x_kv_usage": x_kv_usage,
        "y_context_length": y_context_length,
        "z_itl": z_itl,
        "z_thpt_per_gpu": z_thpt_per_gpu,
        "max_kv_tokens": max_kv_tokens,
    }
    logger.info(f"Decode profiling done: {len(x_kv_usage)} data points")
    return data


def _update_configmap(name: str, namespace: str, prefill_data: dict, decode_data: dict):
    """Create or update a Kubernetes ConfigMap with profiling results."""
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        data={
            "prefill_raw_data.json": json.dumps(prefill_data, indent=2),
            "decode_raw_data.json": json.dumps(decode_data, indent=2),
        },
    )
    try:
        v1.read_namespaced_config_map(name=name, namespace=namespace)
        v1.replace_namespaced_config_map(name=name, namespace=namespace, body=cm)
        logger.info(f"ConfigMap {namespace}/{name} updated")
    except client.ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(namespace=namespace, body=cm)
            logger.info(f"ConfigMap {namespace}/{name} created")
        else:
            raise


def run_profiling(args):
    """Run full SLA profiling pipeline and store results."""
    logger.info("SLA Profiler starting")
    logger.info(f"  BASE_URL: {args.base_url}")
    logger.info(f"  MODEL: {args.model_name}")
    logger.info(f"  NUM_GPUS: {args.num_gpus}")
    logger.info(f"  MAX_KV_TOKENS: {args.max_kv_tokens}")
    logger.info(f"  MAX_CONTEXT_LENGTH: {args.max_context_length}")

    os.makedirs(args.output_dir, exist_ok=True)

    prefill_isl_points = [int(x) for x in args.prefill_isl_points.split(",")]
    decode_isl_points = [int(x) for x in args.decode_isl_points.split(",")]

    # Phase 1: Prefill
    prefill_data = profile_prefill(
        base_url=args.base_url,
        model=args.model_name,
        tokenizer=args.tokenizer,
        num_gpus=args.num_gpus,
        max_context_length=args.max_context_length,
        output_dir=args.output_dir,
        isl_points=prefill_isl_points,
    )

    # Phase 2: Decode
    decode_data = profile_decode(
        base_url=args.base_url,
        model=args.model_name,
        tokenizer=args.tokenizer,
        num_gpus=args.num_gpus,
        max_kv_tokens=args.max_kv_tokens,
        max_context_length=args.max_context_length,
        output_dir=args.output_dir,
        isl_points=decode_isl_points,
        concurrency_granularity=args.decode_granularity,
    )

    # Save locally
    prefill_path = os.path.join(args.output_dir, "prefill_raw_data.json")
    decode_path = os.path.join(args.output_dir, "decode_raw_data.json")
    with open(prefill_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_path, "w") as f:
        json.dump(decode_data, f, indent=2)
    logger.info(f"Results saved to {args.output_dir}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Prefill: {len(prefill_data['prefill_isl'])} points")
    for i, isl in enumerate(prefill_data["prefill_isl"]):
        logger.info(f"  ISL={isl}: TTFT={prefill_data['prefill_ttft'][i]:.1f}ms, thpt/gpu={prefill_data['prefill_thpt_per_gpu'][i]:.0f} tok/s")
    logger.info(f"Decode: {len(decode_data['x_kv_usage'])} points")
    if decode_data["z_thpt_per_gpu"]:
        logger.info(f"  thpt/gpu range: {min(decode_data['z_thpt_per_gpu']):.0f} - {max(decode_data['z_thpt_per_gpu']):.0f} tok/s")
        logger.info(f"  ITL range: {min(decode_data['z_itl']):.1f} - {max(decode_data['z_itl']):.1f} ms")

    # Update ConfigMap
    if args.update_configmap:
        _update_configmap(args.output_configmap, args.namespace, prefill_data, decode_data)

    # Print JSON to stdout for recovery from pod logs
    print("\n===PREFILL_RAW_DATA_JSON_START===")
    print(json.dumps(prefill_data, indent=2))
    print("===PREFILL_RAW_DATA_JSON_END===")
    print("\n===DECODE_RAW_DATA_JSON_START===")
    print(json.dumps(decode_data, indent=2))
    print("===DECODE_RAW_DATA_JSON_END===")

    logger.info("Profiling complete!")
