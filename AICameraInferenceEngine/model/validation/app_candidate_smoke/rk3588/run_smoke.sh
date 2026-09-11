#!/bin/bash
set -euo pipefail
smoke_root=/home/orangepi/vac-app-candidate-smoke-20260910
# Network and mount namespaces isolate the existing application and /model.
exec bwrap --die-with-parent --new-session --unshare-net \
  --tmpfs / \
  --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib \
  --ro-bind /etc /etc --ro-bind /sys /sys --dev-bind /dev /dev --proc /proc \
  --ro-bind "$smoke_root/venv" "$smoke_root/venv" \
  --bind "$smoke_root/app" /app \
  --ro-bind "$smoke_root/app/model" /model \
  --bind "$smoke_root/logs" /logs \
  --bind "$smoke_root/src" /src \
  --bind "$smoke_root/output" /smoke-output \
  --tmpfs /tmp --chdir /app \
  --setenv LD_LIBRARY_PATH /app/libs \
  --setenv PYTHONDONTWRITEBYTECODE 1 \
  /bin/bash -c 'set -a; . configs/candidate-rk3588.env; set +a; exec /home/orangepi/vac-app-candidate-smoke-20260910/venv/bin/python tools/smoke_candidate_app.py --image test_face.jpg --output /smoke-output/result.json'
