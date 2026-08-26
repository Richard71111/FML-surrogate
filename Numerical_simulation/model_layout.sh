#!/usr/bin/env bash

# Shared FEM/GJ naming contract for numerical simulation launchers.

normalize_fem_gj() {
    MESH_IDX="${MESH_IDX:-${FEM_MESH:-${FEM:-1}}}"
    GJ_COUPLING="${GJ_COUPLING:-${GJ:-strong}}"

    if [[ ! "${MESH_IDX}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MESH_IDX/FEM_MESH must be a positive integer; got '${MESH_IDX}'." >&2
        return 2
    fi
    GJ_COUPLING="${GJ_COUPLING,,}"
    case "${GJ_COUPLING}" in
        strong|weak) ;;
        *)
            echo "ERROR: GJ_COUPLING must be strong or weak; got '${GJ_COUPLING}'." >&2
            return 2
            ;;
    esac

    FEM_MESH="${MESH_IDX}"
    FEM="${MESH_IDX}"
    GJ="${GJ_COUPLING}"
    FEM_DIR="FEM${MESH_IDX}"
    GJ_DIR="GJ${GJ_COUPLING}"
    export MESH_IDX FEM_MESH FEM GJ_COUPLING GJ FEM_DIR GJ_DIR
}

derive_model_name_base() {
    case "${MODEL_TYPE,,}" in
        auto|fnn) MODEL_TYPE="FNN" ;;
        resnet) MODEL_TYPE="ResNet" ;;
        *)
            echo "ERROR: MODEL_TYPE must be auto, FNN, or ResNet; got '${MODEL_TYPE}'." >&2
            return 2
            ;;
    esac

    local dt_canonical dt_tag hidden_tag expected
    dt_canonical="$(awk -v dt="${DT_FML}" 'BEGIN { printf "%.15g", dt }')"
    dt_tag="${dt_canonical//./p}"
    hidden_tag="${HIDDEN_LAYERS//[[:space:]]/}"
    hidden_tag="${hidden_tag#[}"
    hidden_tag="${hidden_tag%]}"
    hidden_tag="${hidden_tag//,/-}"
    if [[ -z "${hidden_tag}" ]]; then
        echo "ERROR: HIDDEN_LAYERS is required to derive MODEL_NAME_BASE." >&2
        return 2
    fi

    expected="FEM${MESH_IDX}_GJ${GJ_COUPLING}_VC_${DATASET}_nm${MEMLEN}_nr${NREC}_dt${dt_tag}_${MODEL_TYPE}_hid${hidden_tag}"
    # Do not inherit a stale MODEL_NAME_BASE through ``sbatch --export=ALL``.
    # FEM/GJ and the explicit network settings are the single source of truth.
    MODEL_NAME_BASE="${expected}"
    export MODEL_TYPE MODEL_NAME_BASE
}
