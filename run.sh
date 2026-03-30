#!/usr/bin/env bash
set -euo pipefail

cd /home/user/bot_extract
exec conda run -n bot_env streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8502}"
