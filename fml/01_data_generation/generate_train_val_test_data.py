"""
Traning, validation, and test data generation for surrogate model training and evaluation.

This script generates data by loading the numerical simulation results from: ID_FML_Surrogate/data/dataset
This script serves as the main script run by python3 generate_train_val_test_data

"""

# Import libraries
import os
import sys
import time
import h5py
import numpy as np
import data_processing as dp
import plot

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, PROJECT_ROOT)

from config import Config


# Set global config parameters
cfg = Config()

T_SIM           = cfg.T_SIM                         # Total simulation time (ms)
DT_STEP         = cfg.dt                            # selected FML sampling interval
IGNORE_START    = cfg.IGNORE_START                  # ignore initial steps of each trajectory when sampling bursts (to avoid transients)    
BURST_LEN       = cfg.BURST_LEN                     # total number of samples per burst
VAL_BURST_LEN   = cfg.VAL_BURST_LEN                 # fixed validation rollout length
TEST_BURST_LEN  = cfg.NUM_MEMORY + 1 + int(cfg.T_TEST/cfg.dt)
TEST_NREC       = TEST_BURST_LEN - cfg.NUM_MEMORY - 1
NUM_BURSTS      = cfg.NUM_BURSTS                    # total number of bursts for each trajectory
NUM_BURSTS_VAL  = cfg.NUM_VAL                       # number of bursts for validation per trajectory
NUM_BURSTS_TEST = cfg.NUM_TEST                      # number of bursts for testing per trajectory
NUM_TEST_TRAJECTORIES = cfg.NUM_TEST_TRAJECTORIES   # number of trajectories represented in test
GJ_coupling     = cfg.GJ_COUPLING                   # "strong" or "weak" coupling for data generation   
Save_data       = True                               # Wheter to save the generated train/val/test data (set to False for quick testing, set to True for full data generation)
# File path parameters
source_dir      = cfg.STRONG_PATH if GJ_coupling == "strong" else cfg.WEAK_PATH 
file_name       = f"FEMDATA_{cfg.MESH_IDX}mat_StrongGJ_BCL_{cfg.BCL}.mat"

# Time variable
t_interp        = np.arange(0, T_SIM + 0.5 * DT_STEP, DT_STEP)      # Time points for interpolation
max_show        = 100     # Max number of burst intervals to show in the plot (for quick check)
# Helper functions
def load_h5py_data(file_path):
    """Load data from .mat file using h5py and convert to numpy arrays.

    Returns a dictionary with keys corresponding to dataset names and values as numpy arrays.
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            # Convert to numpy array (h5py returns in row-major order)
            data[key] = np.array(dataset)
    return data

def main():
    # Print header
    print("=" * 60)
    print("Generating training, validation, and test data for surrogate model...")
    
    t_data_start = time.time()
    # Load raw data from the source directory
    mesh_path   = os.path.join(source_dir, file_name)
    data        = load_h5py_data(mesh_path)

    phi_mat     = data['phi']
    Icleft_mat  = data['Icleft']
    t_old       = data['t_save'].squeeze()                 # shape (T,)

    # Orient and interpolate data
    phi_mat     = dp.orient_data_matrix(phi_mat)           # shape (T, Ncell)
    Icleft_mat  = dp.orient_data_matrix(Icleft_mat)        # shape (T, 2, Ncell-1)
    phi_mat     = dp.interpolate_data_matrix(t_old, t_interp, phi_mat)           # shape (len(t_interp), Ncell)
    Icleft_mat  = dp.interpolate_data_matrix(t_old, t_interp, Icleft_mat)        # shape (len(t_interp), 2, Ncell-1)

    # Generate burst
    if phi_mat.shape[0] != Icleft_mat.shape[0]:
        raise ValueError(f"Time dimension mismatch after interpolation: phi_mat.shape[0] = {phi_mat.shape[0]}, Icleft_mat.shape[0] = {Icleft_mat.shape[0]}")
    
    print(f"Data loaded and processed. Time taken: {time.time() - t_data_start:.2f} seconds.")

    trajectory_length = len(t_interp)
    num_trajectories  = Icleft_mat.shape[-1] # Ncell -1
    print("Data summary:")
    print(f"    Samples number per trajectory: {trajectory_length}")
    print(f"    Number of trajectories: {num_trajectories}")
    print(f"    Bursts per trajectory: {NUM_BURSTS}")
    print(f"    Test trajectories: {min(NUM_TEST_TRAJECTORIES, num_trajectories)}")
    print(f"    Burst length Train: {BURST_LEN} (NUM_MEMORY={cfg.NUM_MEMORY}, NUM_RECURRENT={cfg.NUM_RECURRENT})")
    print(f"    Burst length Validation: {VAL_BURST_LEN} "
          f"(nR_val={cfg.NUM_VAL_RECURRENT}, "
          f"{cfg.NUM_VAL_RECURRENT * cfg.dt:g} ms)")
    print(f"    Burst length Test: {TEST_BURST_LEN} "
          f"(nR_test={TEST_NREC}, {cfg.T_TEST} ms)")
    print(f"    Time scales: DT_STEP={DT_STEP}")
    print(f"    GJ coupling: {GJ_coupling}")
    print(f"    Mesh index: {cfg.MESH_IDX}")
    print(f"    Basic cycle length (BCL): {cfg.BCL}")
    print("    Ctrl contract: aligned, one sample longer than QoI")
    print("=" * 60)

    np.random.seed(42)
    seeds_general     = np.random.randint(0, 2**31, size=3)    # seeds for generating seeds of train/val/test split
    np.random.seed(seeds_general[0])
    seeds_list_train  = np.random.randint(0, 2**31, size=num_trajectories)
    np.random.seed(seeds_general[1])
    seeds_list_val    = np.random.randint(0, 2**31, size=num_trajectories)
    np.random.seed(seeds_general[2])
    seeds_list_test   = np.random.randint(0, 2**31, size=num_trajectories)
    test_rng = np.random.default_rng(cfg.DATA_SEED + 1)
    test_trajectory_indices = set(
        test_rng.choice(
            num_trajectories,
            size=min(NUM_TEST_TRAJECTORIES, num_trajectories),
            replace=False,
        ).tolist()
    )

    # Declare lists to hold bursts for all trajectories
    all_phi_train = []
    all_I_train   = []

    all_phi_val   = []
    all_I_val     = []

    all_phi_test  = []
    all_I_test    = []

    t0 = time.time()    # start time for burst generation

    for traj_idx in range(num_trajectories):

        # Generate burst indices for train/val/test split
        burst_idx_train = dp.generate_burst_index(
            trajectory_length,
            BURST_LEN + 1,
            NUM_BURSTS,
            seeds_list_train[traj_idx],
            IGNORE_START
        )

        burst_idx_val = dp.generate_burst_index(
            trajectory_length,
            VAL_BURST_LEN + 1,
            NUM_BURSTS_VAL,
            seeds_list_val[traj_idx],
            IGNORE_START
        )

        if traj_idx in test_trajectory_indices:
            burst_idx_test = dp.generate_burst_index(
                trajectory_length,
                TEST_BURST_LEN + 1,
                NUM_BURSTS_TEST,
                seeds_list_test[traj_idx],
                IGNORE_START
            )
        else:
            burst_idx_test = np.empty(0, dtype=np.int64)

        # ---------------- Train bursts ----------------
        I_traj_train, phi_traj_train = dp.extract_qoi_ctrl_bursts(
            Icleft_mat, phi_mat, burst_idx_train, BURST_LEN, traj_idx,
            channel_first=True
        )

        all_phi_train.append(phi_traj_train)
        all_I_train.append(I_traj_train)

        # ---------------- Validation bursts / segments ----------------
        I_traj_val, phi_traj_val = dp.extract_qoi_ctrl_bursts(
            Icleft_mat, phi_mat, burst_idx_val, VAL_BURST_LEN, traj_idx,
            channel_first=True
        )

        all_phi_val.append(phi_traj_val)
        all_I_val.append(I_traj_val)

        # ---------------- Test bursts / segments ----------------
        if traj_idx in test_trajectory_indices:
            I_traj_test, phi_traj_test = dp.extract_qoi_ctrl_bursts(
                Icleft_mat, phi_mat, burst_idx_test, TEST_BURST_LEN, traj_idx,
                channel_first=True
            )

            all_phi_test.append(phi_traj_test)
            all_I_test.append(I_traj_test)

        elapsed = time.time() - t0
        print(f"Trajectory {traj_idx+1}/{num_trajectories} processed. Time taken: {elapsed:.2f} seconds.")

    # Concatenate bursts from all pair trajectories
    phi_train = np.concatenate(all_phi_train, axis=0)
    I_train   = np.concatenate(all_I_train, axis=0)

    phi_val = np.concatenate(all_phi_val, axis=0)
    I_val   = np.concatenate(all_I_val, axis=0)

    phi_test = np.concatenate(all_phi_test, axis=0)
    I_test   = np.concatenate(all_I_test, axis=0)


    # Convert to float32 to save storage
    phi_train = phi_train.astype(np.float32)
    I_train   = I_train.astype(np.float32)

    phi_val = phi_val.astype(np.float32)
    I_val   = I_val.astype(np.float32)

    phi_test = phi_test.astype(np.float32)
    I_test   = I_test.astype(np.float32)


    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Burst generation complete. Total Time taken: {elapsed:.2f} seconds.")

    print("Generated data shapes:")
    print(f"    phi_train shape: {phi_train.shape}")
    print(f"    I_train shape:   {I_train.shape}")
    print(f"    phi_val shape:   {phi_val.shape}")
    print(f"    I_val shape:     {I_val.shape}")
    print(f"    phi_test shape:  {phi_test.shape}")
    print(f"    I_test shape:    {I_test.shape}")
    
    plot.plot_burst_intervals(
        burst_idx_train=burst_idx_train,
        burst_idx_val=burst_idx_val,
        burst_idx_test=burst_idx_test,
        train_len=BURST_LEN,
        val_len=VAL_BURST_LEN,
        test_len=TEST_BURST_LEN,
        trajectory_length=trajectory_length,
        max_show=max_show,
        title=f"Trajectory {traj_idx + 1}: burst interval check first {max_show} bursts",
        save_path = cfg.EVAL_DIR
    )
    # os.makedirs(cfg.TRAIN_DATA_PATH, exist_ok=True)

    # if Save_data:
    #     np.savez(os.path.join(cfg.TRAIN_DATA_PATH, f"train_data_{GJ_coupling}_BCL_{cfg.BCL}_Mesh_{cfg.MESH_IDX}.npz"), phi=phi_train, I=I_train)
    #     np.savez(os.path.join(cfg.TRAIN_DATA_PATH, f"val_data_{GJ_coupling}_BCL_{cfg.BCL}_Mesh_{cfg.MESH_IDX}.npz"), phi=phi_val, I=I_val)
    #     np.savez(os.path.join(cfg.TRAIN_DATA_PATH, f"test_data_{GJ_coupling}_BCL_{cfg.BCL}_Mesh_{cfg.MESH_IDX}.npz"), phi=phi_test, I=I_test)

    if Save_data:
        np.savez(cfg.TRAIN_FILE, qoi=I_train, ctrl=phi_train,
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(cfg.NUM_RECURRENT),
                 **dp.dataset_metadata(cfg.dt))
        np.savez(cfg.VAL_FILE, qoi=I_val, ctrl=phi_val,
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(cfg.NUM_VAL_RECURRENT),
                 **dp.dataset_metadata(cfg.dt))
        np.savez(cfg.TEST_FILE, qoi=I_test, ctrl=phi_test,
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(TEST_NREC),
                 num_source_trajectories=np.array(len(test_trajectory_indices)),
                 **dp.dataset_metadata(cfg.dt))
if __name__ == "__main__":
    main()
