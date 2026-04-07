#!/bin/bash
# RunPod management helper for Parameter Golf experiments
# Usage: ./runpod.sh [list|start|stop|status|billing|create-1x|create-8x]

API_KEY="${RUNPOD_API_KEY}"
BASE="https://rest.runpod.io/v1"
POD_ID="${RUNPOD_POD_ID}"

if [ -z "$API_KEY" ]; then
  echo "Error: RUNPOD_API_KEY not set. Check ~/.zshrc"
  exit 1
fi

case "$1" in
  list)
    curl -s "$BASE/pods" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    ;;
  start)
    if [ -z "$POD_ID" ]; then echo "Error: Set RUNPOD_POD_ID first"; exit 1; fi
    echo "Starting pod $POD_ID..."
    curl -s -X POST "$BASE/pods/$POD_ID/start" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    ;;
  stop)
    if [ -z "$POD_ID" ]; then echo "Error: Set RUNPOD_POD_ID first"; exit 1; fi
    echo "Stopping pod $POD_ID..."
    curl -s -X POST "$BASE/pods/$POD_ID/stop" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    echo "Pod stopped. Billing paused."
    ;;
  status)
    if [ -z "$POD_ID" ]; then echo "Error: Set RUNPOD_POD_ID first"; exit 1; fi
    curl -s "$BASE/pods/$POD_ID" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    ;;
  billing)
    curl -s "$BASE/billing/pods" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    ;;
  create-1x)
    echo "Creating 1xH100 pod (cheap testing ~\$2.50/hr)..."
    curl -s -X POST "$BASE/pods" \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "imageName": "runpod/pytorch:2.9.1-py3.12-cuda12.8.1-cudnn9.8.0-devel-ubuntu22.04",
        "name": "param-golf-1xh100",
        "cloudType": "COMMUNITY",
        "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
        "gpuCount": 1,
        "containerDiskInGb": 100,
        "volumeInGb": 50,
        "ports": ["22/tcp"]
      }' | python3 -m json.tool
    echo ""
    echo "Save the pod ID and run: export RUNPOD_POD_ID=<id>"
    ;;
  create-8x)
    echo "Creating 8xH100 pod (final validation ~\$20/hr)..."
    curl -s -X POST "$BASE/pods" \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "imageName": "runpod/pytorch:2.9.1-py3.12-cuda12.8.1-cudnn9.8.0-devel-ubuntu22.04",
        "name": "param-golf-8xh100",
        "cloudType": "SECURE",
        "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
        "gpuCount": 8,
        "containerDiskInGb": 100,
        "volumeInGb": 50,
        "ports": ["22/tcp"]
      }' | python3 -m json.tool
    echo ""
    echo "Save the pod ID and run: export RUNPOD_POD_ID=<id>"
    ;;
  delete)
    if [ -z "$POD_ID" ]; then echo "Error: Set RUNPOD_POD_ID first"; exit 1; fi
    read -p "WARNING: This permanently deletes pod $POD_ID. Type 'yes' to confirm: " confirm
    if [ "$confirm" = "yes" ]; then
      curl -s -X DELETE "$BASE/pods/$POD_ID" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
      echo "Pod deleted."
    else
      echo "Aborted."
    fi
    ;;
  *)
    echo "RunPod helper for Parameter Golf"
    echo ""
    echo "Usage: ./runpod.sh <command>"
    echo ""
    echo "Commands:"
    echo "  list       List all pods"
    echo "  create-1x  Create 1xH100 pod (~\$2.50/hr, for testing)"
    echo "  create-8x  Create 8xH100 pod (~\$20/hr, for validation)"
    echo "  start      Start a stopped pod"
    echo "  stop       Stop pod (billing pauses, /workspace persists)"
    echo "  status     Get pod details"
    echo "  billing    Check pod billing"
    echo "  delete     Permanently delete pod"
    echo ""
    echo "Set RUNPOD_POD_ID after creating a pod:"
    echo "  export RUNPOD_POD_ID=<id>"
    ;;
esac
