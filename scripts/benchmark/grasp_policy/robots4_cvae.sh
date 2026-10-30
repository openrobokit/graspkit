#!/bin/bash
# The ARX diverse CVAE recipe on another embodiment's official 1007 data (depth), with the official evaluator and
# test: train the CVAE actor on history4096's positives bank, re-encode its 256-proposal replay with the actor,
# train the evaluator on those labels, then score v0.5 test500 open loop (learned selector, guidance 8).
#   robots4_cvae.sh ROBOT train            # actor -> evaluator -> $O/policy.pt
#   robots4_cvae.sh ROBOT eval RANK WORLD  # one shard of the 500 test scenes -> $O/eval/open.rank$RANK.jsonl
# BANK (the official run's root), OUT (outputs, default /fleet-state/robots4), ITERS (16000), SCENES (test scenes),
# MAX_SCENES (per shard, smoke tests); train: KL (CVAE kl_weight, default 0.1), LOCAL=1 copies the bank to /tmp first
# (cephfs reads otherwise dominate: ~9 min to cache the actor data, ~14 min to re-encode the replay);
# eval: POLICY (default $O/policy.pt) and TAG (default open) score another checkpoint into $O/eval/$TAG.rank*; each shard
# also saves its executed joint trajectories ($TAG.traj.rank$RANK.pt).
set -euo pipefail
R=$1; STAGE=$2
BANK=${BANK:-/fleet-state/unified-policy/benchmark/official-rig-recipe-1007}/$R/history4096
O=${OUT:-/fleet-state/robots4}/$R
ITERS=${ITERS:-16000}
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p "$O"
case $STAGE in
train)
  if [ "${LOCAL:-0}" = 1 ]; then
    T=/tmp/robots4_bank/$R; mkdir -p "$T"
    for f in train4096.h5 scenes.msgpack replay4096.pt observations4096.h5; do [ -f "$T/$f" ] || cp "$BANK/$f" "$T/$f"; done
    BANK=$T
  fi
  NAME=${R}_unified_depth__cvae${TAG:+_$TAG}  # TAG keeps concurrent runs' logs/train dirs apart
  if [ ! -f "$O/actor.pt" ]; then
    uv run --frozen scripts/stages/train/train.py "${R}_unified_depth" --name "$NAME" \
      --processed_data_path "$BANK/train4096.h5" --scene_description_path "$BANK/scenes.msgpack" \
      --dataset.num_actions 64 \
      --trainer.batch_size 128 --trainer.lr 0.0003 --trainer.total_iters "$ITERS" --trainer.save_freq "$ITERS" \
      agent:grasp_policy_config --agent.generative_model cvae --agent.kl_weight "${KL:-0.1}" \
      > "$O/train.log" 2>&1
    cp "logs/train/$NAME/checkpoints/iter_$ITERS.pt" "$O/actor.pt.tmp" && mv "$O/actor.pt.tmp" "$O/actor.pt"
  fi
  [ -f "$O/replay.pt" ] || uv run --frozen scripts/benchmark/grasp_policy/reencode_replay.py --checkpoint "$O/actor.pt" \
    --replay "$BANK/replay4096.pt" --observations "$BANK/observations4096.h5" --scenes "$BANK/scenes.msgpack" \
    --output "$O/replay.pt" > "$O/reencode.log" 2>&1
  [ -f "$O/policy.pt" ] || uv run --frozen scripts/benchmark/grasp_policy/train_critic_scenes.py --scenes "$BANK/scenes.msgpack" \
    --experiment "${R}_unified_depth" --replay "$O/replay.pt" --checkpoint "$O/actor.pt" --output "$O/policy.pt" \
    --ranking-weight 1 > "$O/critic.log" 2>&1
  ;;
eval)
  K=$3; W=$4; mkdir -p "$O/eval"
  SCENES=${SCENES:-$HOME/.graspkit/data/scene_description/dex1b/v0.5.0/tabletop_mini.msgpack}
  D=$HOME/.graspkit/data/scene_description/dex1b/v0.5.0  # the experiments' own scene file (the source splits)
  [ -f "$D/tabletop_mini.msgpack" ] || { mkdir -p "$D" && cp "$SCENES" "$D/tabletop_mini.msgpack"; }
  F="$O/eval/${TAG:-open}.rank$K.jsonl"
  [ -f "$F.complete" ] && exit 0
  rm -f "$F"
  uv run --frozen scripts/benchmark/grasp_policy/evaluate_policy.py --checkpoint "${POLICY:-$O/policy.pt}" --mode policy_open_loop \
    --selector learned --guidance-steps 8 --num-samples 64 --split test \
    --scene-description-path "$SCENES" \
    --rank "$K" --world-size "$W" --max-scenes "${MAX_SCENES:-0}" --output "$F" --trajectories "$O/eval/${TAG:-open}.traj.rank$K.pt" > "$F.log" 2>&1
  touch "$F.complete"
  ;;
esac
