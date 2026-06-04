#!/bin/bash
# Port-forward Grafana to localhost:3000
echo "Forwarding Grafana to http://localhost:3000 (admin/admin)"
echo "Dashboard: Dashboards -> RBG Planner Dashboard"
echo "Press Ctrl+C to stop"
kubectl port-forward svc/grafana -n demo 3000:3000
