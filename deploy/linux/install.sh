#!/usr/bin/env bash
# Life Guard installer for Linux and WSL2 (Ubuntu 22.04+ / Debian 12+), the counterpart of deploy/windows/install.ps1.
#
#   sudo deploy/linux/install.sh [--data-dir /var/lib/lifeguard] [--user lifeguard] [--api-port 8000]
#                                [--admin-user NAME] [--enable]
#
# Steps: 1 service user  2 venv from requirements/serving.lock (--require-hashes)  3 model check (verify_models)
#        4 env file (0600)  5 owner-only server keys (`lifeguard keys init`)  6 migrations (+ first admin, prompted)
#        7 systemd units - installed but NOT enabled: nothing of Life Guard starts at boot unless --enable is given.
# GPU: TensorRT / CUDA come from the NVIDIA driver + the nvidia-*-cu12 wheels; on WSL2 install the Windows NVIDIA
# driver only (never a Linux driver inside WSL). Without a GPU, LIFEGUARD_INFERENCE_PROVIDER=auto runs on
# OpenVINO / CPU and live admission refuses the cameras the CPU cannot analyse at 15 fps.
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR=/var/lib/lifeguard
SVC_USER=lifeguard
API_PORT=8000
ADMIN_USER=""
ENABLE=0
PYTHON="${PYTHON:-python3.12}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --user) SVC_USER="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --admin-user) ADMIN_USER="$2"; shift 2 ;;
    --enable) ENABLE=1; shift ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run with sudo (creates a service user and systemd units)" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found (Life Guard needs Python 3.12)" >&2; exit 1; }
VENV="$INSTALL_DIR/build/service/venv"
step() { echo; echo "== $*"; }
as_svc() { runuser -u "$SVC_USER" -- env LIFEGUARD_ENV_FILE="$DATA_DIR/lifeguard.env" "$@"; }

# Rename shim: an untouched AquaGuard install's data is at /var/lib/aquaguard under the "aquaguard" user. If  # legacy-name
# --data-dir was not given and the new directory does not exist yet, but the old one does, keep using it (nothing
# is moved or copied) so an in-place upgrade does not strand an existing install's database and keys.
if [[ "$DATA_DIR" == /var/lib/lifeguard && ! -d "$DATA_DIR" && -d /var/lib/aquaguard ]]; then
  echo "using the existing data directory /var/lib/aquaguard (move it to $DATA_DIR when convenient)" >&2
  DATA_DIR=/var/lib/aquaguard
  id -u aquaguard >/dev/null 2>&1 && SVC_USER=aquaguard
fi

step "0 remove retired AquaGuard systemd units, if any"  # legacy-name
for unit in aquaguard-api aquaguard-supervisor aquaguard-upload-worker; do  # legacy-name
  if systemctl list-unit-files "$unit.service" >/dev/null 2>&1 && [[ -f "/etc/systemd/system/$unit.service" ]]; then
    systemctl stop "$unit" 2>/dev/null || true
    systemctl disable "$unit" 2>/dev/null || true
    rm -f "/etc/systemd/system/$unit.service"
    echo "removed retired unit $unit"
  fi
done

step "1 service user $SVC_USER"
id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SVC_USER"
install -d -m 0700 -o "$SVC_USER" -g "$SVC_USER" "$DATA_DIR"

step "2 venv from requirements/serving.lock (--require-hashes)"
[[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --require-hashes --no-deps \
  -r "$INSTALL_DIR/requirements/serving.lock"

step "3 models (sha256 against backend/models/MODELS.json)"
"$VENV/bin/python" "$INSTALL_DIR/tools/verify_models.py" || {
  echo "model check failed: export the pose model (tools/export_rtmpose.py --register) and copy the weights" >&2
  exit 1
}

step "4 env file $DATA_DIR/lifeguard.env (0600, owner $SVC_USER)"
if [[ ! -f "$DATA_DIR/lifeguard.env" ]]; then
  umask 077
  cat > "$DATA_DIR/lifeguard.env" <<ENV
LIFEGUARD_ENV=production
LIFEGUARD_DATA_DIR=$DATA_DIR
LIFEGUARD_INFERENCE_PROVIDER=auto
ENV
  chown "$SVC_USER:$SVC_USER" "$DATA_DIR/lifeguard.env"
fi

step "5 server keys (owner-only files, mode 0600)"
if [[ ! -f "$DATA_DIR/keys/session.key" ]]; then
  (cd "$INSTALL_DIR/backend" && as_svc "$VENV/bin/python" -m lifeguard keys init)
fi

step "6 database migrations"
(cd "$INSTALL_DIR/backend" && as_svc "$VENV/bin/python" -m lifeguard db upgrade)
if [[ -n "$ADMIN_USER" ]]; then
  (cd "$INSTALL_DIR/backend" && as_svc "$VENV/bin/python" -m lifeguard admin create --username "$ADMIN_USER")
fi

step "7 systemd units (installed; enabled only with --enable)"
for unit in lifeguard-api lifeguard-supervisor; do
  sed -e "s#@USER@#$SVC_USER#g" -e "s#@INSTALL_DIR@#$INSTALL_DIR#g" -e "s#@DATA_DIR@#$DATA_DIR#g" \
      -e "s#@VENV@#$VENV#g" -e "s#@API_PORT@#$API_PORT#g" \
      "$INSTALL_DIR/deploy/linux/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
if [[ $ENABLE -eq 1 ]]; then
  systemctl enable lifeguard-api lifeguard-supervisor
fi
echo
echo "done. Start:  sudo systemctl start lifeguard-api lifeguard-supervisor"
echo "      Stop:   sudo systemctl stop lifeguard-supervisor lifeguard-api"
echo "      GPU:    sudo -u $SVC_USER $VENV/bin/python $INSTALL_DIR/tools/build_trt_engines.py --data-dir $DATA_DIR"
