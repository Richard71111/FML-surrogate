#!/usr/bin/env bash
#
# Submit the complete structured model sweep:
#   3 FEM meshes x 2 GJ couplings x 15 FML shapes = 90 simulations.
#
# This is a launcher, not a Slurm job. Run it with bash on a login node.
#
# Base checkpoints:
#   bash Numerical_simulation/submit_all_90_numerical_simulations.sh
#
# Fine-tuned checkpoints:
#   FINETUNE=true FINETUNE_BCL=1000 \
#     bash Numerical_simulation/submit_all_90_numerical_simulations.sh
#
# Every run is compared over exactly five beats with the matching FEM mesh and
# GJ coupling in MultiMesh_Cable_Cleft/VaryBCL/{Strong,Weak}GJ/BCL200.
#
# Validate all 90 jobs without submitting:
#   DRY_RUN=true \
#     bash Numerical_simulation/submit_all_90_numerical_simulations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FAMILY_LAUNCHER="${SCRIPT_DIR}/submit_all_numerical_simulations.sh"

MESHES=(10 102 133)
GJ_COUPLINGS=(strong weak)
EXPECTED_PER_FAMILY=15
EXPECTED_TOTAL=90

FINETUNE="${FINETUNE:-true}"
FINETUNE_BCL="${FINETUNE_BCL:-1000}"
FINETUNE_MODEL="${FINETUNE_MODEL:-best_val}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_val}"
DRY_RUN="${DRY_RUN:-false}"
BCL="${BCL:-200}"
NBEATS="${NBEATS:-5}"
COMPARE_NBEATS=5
RECORD_LAST_BEAT_ONLY=false
COMPARE_TO_MATLAB=true
STIM_AMP="${STIM_AMP:-50}"
FAST_RUN="${FAST_RUN:-true}"
REQUIRE_CHECKPOINT_COMPLETE="${REQUIRE_CHECKPOINT_COMPLETE:-true}"
MATLAB_REFERENCE_ROOT="${MATLAB_REFERENCE_ROOT:-/fs/ess/PAS1622/RichardSui/MultiMesh_Cable_Cleft/VaryBCL}"

case "${FINETUNE,,}" in
    true|false) FINETUNE="${FINETUNE,,}" ;;
    *) echo "ERROR: FINETUNE must be true or false, got '${FINETUNE}'." >&2; exit 2 ;;
esac
case "${DRY_RUN,,}" in
    true|false) DRY_RUN="${DRY_RUN,,}" ;;
    *) echo "ERROR: DRY_RUN must be true or false, got '${DRY_RUN}'." >&2; exit 2 ;;
esac
if [[ "${BCL}" != "200" ]]; then
    echo "ERROR: this 90-model comparison launcher requires BCL=200; got ${BCL}." >&2
    exit 2
fi
if [[ "${NBEATS}" != "5" ]]; then
    echo "ERROR: this 90-model comparison launcher requires NBEATS=5; got ${NBEATS}." >&2
    exit 2
fi

if [[ ! -f "${FAMILY_LAUNCHER}" ]]; then
    echo "ERROR: family launcher not found: ${FAMILY_LAUNCHER}" >&2
    exit 2
fi

reference_for() {
    local mesh="$1"
    local gj="$2"
    local gj_label
    case "${gj}" in
        strong) gj_label="Strong" ;;
        weak) gj_label="Weak" ;;
        *) echo "ERROR: unsupported GJ coupling '${gj}'." >&2; return 2 ;;
    esac
    printf '%s/%sGJ/BCL%s/FEMDATA_%smat_%sGJ_BCL_%s.mat' \
        "${MATLAB_REFERENCE_ROOT}" "${gj_label}" "${BCL}" \
        "${mesh}" "${gj_label}" "${BCL}"
}

for mesh in "${MESHES[@]}"; do
    for gj in "${GJ_COUPLINGS[@]}"; do
        reference="$(reference_for "${mesh}" "${gj}")"
        if [[ ! -f "${reference}" ]]; then
            echo "ERROR: MATLAB reference not found for FEM${mesh}/GJ${gj}: ${reference}" >&2
            exit 2
        fi
    done
done

cd "${PROJECT_ROOT}"
export FINETUNE FINETUNE_BCL FINETUNE_MODEL CHECKPOINT_KIND
export BCL NBEATS COMPARE_NBEATS RECORD_LAST_BEAT_ONLY COMPARE_TO_MATLAB
export STIM_AMP FAST_RUN REQUIRE_CHECKPOINT_COMPLETE

echo "================================================================"
echo "Complete numerical model sweep"
echo "families        : FEM{10,102,133} x GJ{strong,weak}"
echo "expected jobs   : ${EXPECTED_TOTAL}"
echo "fine tune       : ${FINETUNE} (BCL=${FINETUNE_BCL}, ${FINETUNE_MODEL})"
echo "checkpoint kind : ${CHECKPOINT_KIND}"
echo "BCL / nbeats    : ${BCL:-200} / ${NBEATS:-5}"
echo "compare beats   : ${COMPARE_NBEATS}"
echo "stimulus amp    : ${STIM_AMP} (MATLAB protocol)"
echo "fast run        : ${FAST_RUN}"
echo "require complete: ${REQUIRE_CHECKPOINT_COMPLETE}"
echo "reference root  : ${MATLAB_REFERENCE_ROOT}"
echo "dry run         : ${DRY_RUN}"
echo "================================================================"

# Preflight every family before the first real sbatch call. This verifies that
# all logs are readable, configurations are complete, and all selected
# checkpoints exist. It prevents a missing model in a later family from
# producing an avoidable partial 90-model sweep.
total=0
for mesh in "${MESHES[@]}"; do
    for gj in "${GJ_COUPLINGS[@]}"; do
        echo "Preflight FEM${mesh}/GJ${gj}..."
        reference="$(reference_for "${mesh}" "${gj}")"
        output="$(
            MESH_IDX="${mesh}" GJ_COUPLING="${gj}" DRY_RUN=true \
                MATLAB_REFERENCE_MAT="${reference}" \
                bash "${FAMILY_LAUNCHER}"
        )"
        count="$(
            awk -F': ' '/Jobs that would be submitted:/ {value=$2} END {print value}' \
                <<<"${output}"
        )"
        if [[ ! "${count}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${output}" >&2
            echo "ERROR: could not determine the dry-run job count for FEM${mesh}/GJ${gj}." >&2
            exit 3
        fi
        if (( count != EXPECTED_PER_FAMILY )); then
            printf '%s\n' "${output}" >&2
            echo "ERROR: FEM${mesh}/GJ${gj} has ${count} models; expected ${EXPECTED_PER_FAMILY}." >&2
            exit 3
        fi
        printf '  validated: %d models\n' "${count}"
        ((total += count))
    done
done

if (( total != EXPECTED_TOTAL )); then
    echo "ERROR: preflight found ${total} models; expected ${EXPECTED_TOTAL}." >&2
    exit 3
fi
echo "Preflight complete: ${total}/${EXPECTED_TOTAL} models validated."

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=true: no Slurm jobs were submitted."
    exit 0
fi

submitted_families=0
for mesh in "${MESHES[@]}"; do
    for gj in "${GJ_COUPLINGS[@]}"; do
        echo "Submitting FEM${mesh}/GJ${gj} (${EXPECTED_PER_FAMILY} jobs)..."
        reference="$(reference_for "${mesh}" "${gj}")"
        MESH_IDX="${mesh}" GJ_COUPLING="${gj}" DRY_RUN=false \
            MATLAB_REFERENCE_MAT="${reference}" \
            bash "${FAMILY_LAUNCHER}"
        ((submitted_families += 1))
    done
done

echo "================================================================"
echo "Submitted ${EXPECTED_TOTAL} numerical simulations in ${submitted_families} families."
echo "================================================================"
