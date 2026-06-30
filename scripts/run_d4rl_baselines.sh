#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-d4rl_baselines_result}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_DIR}/experiments}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
SEEDS_STR="${SEEDS:-1}"
AGENT_FILES_STR="${AGENT_FILES:-agents/bc.py agents/iql.py agents/ifql.py agents/fql.py agents/fbrac.py agents/gfp.py agents/meanflow_fql.py agents/qam.py agents/rebrac.py agents/shortcut_fql.py}"
D4RL_ENVS_STR="${D4RL_ENVS:-pen-human-v1 pen-cloned-v1 pen-expert-v1 door-human-v1 door-cloned-v1 door-expert-v1 hammer-human-v1 hammer-cloned-v1 hammer-expert-v1 relocate-human-v1 relocate-cloned-v1 relocate-expert-v1}"
EXTRA_FLAGS_STR="${EXTRA_FLAGS:-}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a AGENT_FILES <<< "$AGENT_FILES_STR"
IFS=' ' read -r -a D4RL_ENVS <<< "$D4RL_ENVS_STR"
IFS=' ' read -r -a EXTRA_FLAGS <<< "$EXTRA_FLAGS_STR"

mkdir -p "$SAVE_ROOT"
mkdir -p "$LOG_DIR"

COMMON_FLAGS=(
  "--save_dir=${SAVE_ROOT}"
  "--enable_wandb=0"
  "--wandb_mode=disabled"
  "--wandb_no_local_files=1"
  "--video_episodes=0"
)

FAILED_JOBS=()

check_python() {
  local preflight_log="${LOG_DIR}/_python_preflight.log"
  echo "[$(date '+%F %T')] Python command: ${PYTHON_BIN}"
  if ! "$PYTHON_BIN" -c "import sys; print(sys.executable)" > "$preflight_log" 2>&1; then
    echo "Python preflight failed. Set PYTHON_BIN to the Python executable in your training environment."
    echo "Example:"
    echo "  PYTHON_BIN=/path/to/python bash scripts/run_d4rl_baselines.sh"
    echo "Preflight log: ${preflight_log}"
    tail -n 40 "$preflight_log"
    exit 1
  fi
  echo "[$(date '+%F %T')] Using Python: $(cat "$preflight_log")"
}

agent_name_from_file() {
  local file="$1"
  basename "$file" .py
}

run_exp() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo "[$(date '+%F %T')] START ${name}"
  echo "[$(date '+%F %T')] START ${name}" > "$log_file"

  if ! "$PYTHON_BIN" main.py "$@" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" >> "$log_file" 2>&1; then
    echo "[$(date '+%F %T')] FAIL  ${name}  log=${log_file}"
    echo "[$(date '+%F %T')] FAIL  ${name}" >> "$log_file"
    tail -n 40 "$log_file"
    FAILED_JOBS+=("$name")
    return 1
  fi

  echo "[$(date '+%F %T')] DONE  ${name}  log=${log_file}"
  echo "[$(date '+%F %T')] DONE  ${name}" >> "$log_file"
  return 0
}

check_python

for seed in "${SEED_LIST[@]}"; do
  for agent_file in "${AGENT_FILES[@]}"; do
    if [[ ! -f "$agent_file" ]]; then
      echo "[$(date '+%F %T')] SKIP missing agent file: ${agent_file}"
      FAILED_JOBS+=("missing-${agent_file}")
      continue
    fi

    agent_name="$(agent_name_from_file "$agent_file")"
    if [[ "$agent_name" == "cdp" ]]; then
      echo "[$(date '+%F %T')] SKIP excluded agent: ${agent_file}"
      continue
    fi

    for env_name in "${D4RL_ENVS[@]}"; do
      run_exp "${agent_name}-${env_name}-seed${seed}" \
        --seed="${seed}" \
        --env_name="${env_name}" \
        --wandb_run_group="${agent_name}" \
        --agent="${agent_file}" || true
    done
  done
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  echo "Logs are under: ${LOG_DIR}"
  exit 1
fi

echo "All non-CDP baseline D4RL runs finished."
echo "Experiment outputs: ${SAVE_ROOT}"
echo "Command logs: ${LOG_DIR}"
