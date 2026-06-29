#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-baseline_result}"
SEEDS_STR="${SEEDS:-1}"
TASK_IDS_STR="${TASK_IDS:-task1 task2 task3 task4 task5}"
RUN_OGBENCH="${RUN_OGBENCH:-1}"
RUN_D4RL="${RUN_D4RL:-1}"
RUN_ONLINE="${RUN_ONLINE:-1}"
ONLINE_STEPS="${ONLINE_STEPS:-1000000}"

AGENT_FILES_STR="${AGENT_FILES:-agents/cdp.py agents/codac.py agents/fbrac.py agents/fql.py agents/shortcut_fql.py agents/meanflow_fql.py agents/gfp.py agents/ifql.py agents/iql.py agents/qam.py agents/rebrac.py}"
D4RL_ENVS_STR="${D4RL_ENVS:-pen-human-v1 pen-cloned-v1 pen-expert-v1 door-human-v1 door-cloned-v1 door-expert-v1 hammer-human-v1 hammer-cloned-v1 hammer-expert-v1 relocate-human-v1 relocate-cloned-v1 relocate-expert-v1}"
ONLINE_ENVS_STR="${ONLINE_ENVS:-antmaze-large-navigate-singletask-task1-v0 humanoidmaze-medium-navigate-singletask-task1-v0 cube-double-play-singletask-task2-v0 cube-triple-play-singletask-task1-v0 puzzle-4x4-play-singletask-task4-v0 scene-play-singletask-task2-v0}"
EXTRA_FLAGS_STR="${EXTRA_FLAGS:-}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"
IFS=' ' read -r -a AGENT_FILES <<< "$AGENT_FILES_STR"
IFS=' ' read -r -a D4RL_ENVS <<< "$D4RL_ENVS_STR"
IFS=' ' read -r -a ONLINE_ENVS <<< "$ONLINE_ENVS_STR"
IFS=' ' read -r -a EXTRA_FLAGS <<< "$EXTRA_FLAGS_STR"

mkdir -p "$SAVE_ROOT"
mkdir -p "$SAVE_ROOT/logs"

COMMON_FLAGS=(
  "--save_dir=${SAVE_ROOT}"
  "--enable_wandb=0"
  "--wandb_mode=disabled"
  "--wandb_no_local_files=1"
  "--video_episodes=0"
)

FAILED_JOBS=()

agent_name_from_file() {
  local file="$1"
  basename "$file" .py
}

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  if ! "$PYTHON_BIN" main.py "$@" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}"; then
    echo "[$(date '+%F %T')] FAIL  ${name}"
    FAILED_JOBS+=("$name")
    return 1
  fi
  echo "[$(date '+%F %T')] DONE  ${name}"
  return 0
}

run_agent_env() {
  local agent_file="$1"
  local env_name="$2"
  local seed="$3"
  shift 3

  if [[ ! -f "$agent_file" ]]; then
    echo "[$(date '+%F %T')] SKIP missing agent file: ${agent_file}"
    FAILED_JOBS+=("missing-${agent_file}")
    return 1
  fi

  local agent_name
  agent_name="$(agent_name_from_file "$agent_file")"
  run_exp "${agent_name}-${env_name}-seed${seed}" \
    --seed="${seed}" \
    --env_name="${env_name}" \
    --wandb_run_group="${agent_name}" \
    --agent="${agent_file}" \
    "$@" || true
}

OGBENCH_ENVS=()
if [[ "$RUN_OGBENCH" == "1" ]]; then
  for task_id in "${TASK_IDS[@]}"; do
    OGBENCH_ENVS+=(
      "cube-double-play-singletask-${task_id}-v0"
      "cube-triple-play-singletask-${task_id}-v0"
      "puzzle-3x3-play-singletask-${task_id}-v0"
      "puzzle-4x4-play-singletask-${task_id}-v0"
      "scene-play-singletask-${task_id}-v0"
    )
  done
fi

for seed in "${SEED_LIST[@]}"; do
  for agent_file in "${AGENT_FILES[@]}"; do
    for env_name in "${OGBENCH_ENVS[@]}"; do
      run_agent_env "$agent_file" "$env_name" "$seed"
    done

    if [[ "$RUN_D4RL" == "1" ]]; then
      for env_name in "${D4RL_ENVS[@]}"; do
        run_agent_env "$agent_file" "$env_name" "$seed"
      done
    fi

    if [[ "$RUN_ONLINE" == "1" ]]; then
      for env_name in "${ONLINE_ENVS[@]}"; do
        run_agent_env "$agent_file" "$env_name" "$seed" \
          --online_steps="${ONLINE_STEPS}"
      done
    fi
  done
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All requested baseline experiments finished."
