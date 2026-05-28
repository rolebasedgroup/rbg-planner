"""Core profiling logic.

TODO: Implement SLA profiling pipeline:
1. Query target RBG to discover engine endpoints
2. Run benchmarks at various concurrency levels and sequence lengths
3. Measure TTFT, ITL, throughput per GPU
4. Generate prefill_raw_data.json and decode_raw_data.json
5. Create Kubernetes ConfigMap with profiling results
"""

import json
import logging
import os

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

logger = logging.getLogger(__name__)


def run_profiling(args):
    """Run SLA profiling and store results in a ConfigMap."""
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    logger.info(f"Starting profiling for model={args.model_name} engine={args.engine}")
    logger.info(f"SLA targets: TTFT={args.ttft_sla}ms ITL={args.itl_sla}ms")
    logger.info(f"Target RBG: {args.namespace}/{args.rbg_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    # TODO: Implement actual profiling logic
    # For now, generate placeholder profiling data
    prefill_data = {
        "prefill_isl": [128, 256, 512, 1024, 2048],
        "prefill_ttft": [0.008, 0.015, 0.030, 0.060, 0.120],
        "prefill_thpt_per_gpu": [6000, 5000, 4000, 3000, 2000],
    }
    decode_data = {
        "x_kv_usage": [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90],
        "y_context_length": [350, 350, 700, 700, 1400, 1400, 2800, 2800],
        "z_itl": [0.006, 0.008, 0.009, 0.012, 0.017, 0.022, 0.035, 0.055],
        "z_thpt_per_gpu": [1500, 1350, 1250, 1050, 750, 550, 400, 250],
        "max_kv_tokens": 32768,
    }

    prefill_path = os.path.join(args.output_dir, "prefill_raw_data.json")
    decode_path = os.path.join(args.output_dir, "decode_raw_data.json")

    with open(prefill_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_path, "w") as f:
        json.dump(decode_data, f, indent=2)

    logger.info(f"Profiling data written to {args.output_dir}")

    # Create ConfigMap with profiling results
    _create_configmap(
        name=args.output_configmap,
        namespace=args.namespace,
        prefill_data=json.dumps(prefill_data),
        decode_data=json.dumps(decode_data),
    )
    logger.info(f"Created ConfigMap {args.namespace}/{args.output_configmap}")


def _create_configmap(name: str, namespace: str, prefill_data: str, decode_data: str):
    """Create or update a ConfigMap with profiling results."""
    v1 = client.CoreV1Api()
    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        data={
            "prefill_raw_data.json": prefill_data,
            "decode_raw_data.json": decode_data,
        },
    )
    try:
        v1.read_namespaced_config_map(name=name, namespace=namespace)
        v1.replace_namespaced_config_map(name=name, namespace=namespace, body=cm)
    except client.ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(namespace=namespace, body=cm)
        else:
            raise
