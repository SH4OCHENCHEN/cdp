#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-four_agents_non_visual_result}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_DIR}/experiments}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
SEEDS_STR="${SEEDS:-1}"
TASK_IDS_STR="${TASK_IDS:-task1 task2 task3 task4 task5}"
AGENT_FILES_STR="${AGENT_FILES:-agents/shortcut_fql.py agents/meanflow_fql.py agents/qam.py agents/gfp.py}"
ENV_PREFIXES_STR="${ENV_PREFIXES:-cube-double-play cube-triple-play puzzle-3x3-play puzzle-4x4-play scene-play}"
EXTRA_FLAGS_STR="${EXTRA_FLAGS:-}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"
IFS=' ' read -r -a AGENT_FILES <<< "$AGENT_FILES_STR"
IFS=' ' read -r -a ENV_PREFIXES <<< "$ENV_PREFIXES_STR"
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
    echo "  PYTHON_BIN=/path/to/python bash scripts/run_four_agents_non_visual_tasks.sh"
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
    for env_prefix in "${ENV_PREFIXES[@]}"; do
      for task_id in "${TASK_IDS[@]}"; do
        env_name="${env_prefix}-singletask-${task_id}-v0"
        run_exp "${agent_name}-${env_name}-seed${seed}" \
          --seed="${seed}" \
          --env_name="${env_name}" \
          --wandb_run_group="${agent_name}" \
          --agent="${agent_file}" || true
      done
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

echo "All four-agent non-visual task runs finished."
echo "Experiment outputs: ${SAVE_ROOT}"
echo "Command logs: ${LOG_DIR}"
