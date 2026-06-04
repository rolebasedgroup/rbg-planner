# Demo Video Subtitles

## Scene 1 — Create AutoScaler

Create an AutoScaler CR. It integrates the Dynamo Planner for SLA-driven Prefill/Decode autoscaling. Configure SLA targets, replica bounds, and prediction method.

## Scene 2 — Start Traffic

Send real production traffic to verify autoscaling.

## Scene 3 — Dashboard

Open the observability dashboard. The first row is an overview. The second row shows observed versus predicted request characteristics. The third row shows scaling decisions and latency metrics.

## Scene 4 — Prediction-Driven Scale-Up

Through time-series prediction and throughput profiling, predict next interval's load, calculate required replicas, and scale accordingly.

## Scene 5 — SLA Under Control

As load rises, latency stays within SLA targets thanks to precise scaling.

## Scene 6 — Scale Down

When traffic drops, the AutoScaler detects it and scales down to save cost.