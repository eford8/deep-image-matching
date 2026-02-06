
import subprocess
import os
import glob
from pathlib import Path

# ----------------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------------
COLMAP_BIN = r"C:\Users\forde\Downloads\colmap-x64-windows-cuda\COLMAP.bat"  # path to COLMAP.bat
WORKSPACE = r"C:\Users\forde\PycharmProjects\PythonProject\2d_to3d\deep-image-matching\assets\emi_penguin"
RESULTS = r"results_roma_matching_lowres_quality_high_0043"

IMAGES_DIR = os.path.join(WORKSPACE, "images")
DATABASE_PATH = os.path.join(WORKSPACE, r"database.db")
SPARSE_DIR = os.path.join(WORKSPACE, fr"{RESULTS}\reconstruction")
DENSE_DIR = os.path.join(WORKSPACE, "dense")
DENSE_EE_DIR =  os.path.join(WORKSPACE, RESULTS, "dense_ee")

os.makedirs(SPARSE_DIR, exist_ok=True)
os.makedirs(DENSE_DIR, exist_ok=True)
os.makedirs(DENSE_EE_DIR, exist_ok=True)

# ----------------------------------
# UTILITIES
# ----------------------------------
def run_cmd(cmd, cwd=None):
    """Run a command, print stdout/stderr, and raise with full context if it fails."""
    print(f"\n[RUN] {' '.join(cmd)}\n")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        shell=False  # run .bat directly
    )
    if result.stdout:
        print("[STDOUT]\n", result.stdout)
    if result.stderr:
        print("[STDERR]\n", result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result

def latest_sparse_model_dir(model_num=None):
    """COLMAP mapper writes models to sparse/0, sparse/1, ... Pick the latest."""
    candidates = []
    if os.path.isdir(SPARSE_DIR):
        for d in os.listdir(SPARSE_DIR):
            if d.isdigit() and os.path.isdir(os.path.join(SPARSE_DIR, d)):
                candidates.append(int(d))
    if not candidates:
        raise FileNotFoundError(f"No sparse model found in {SPARSE_DIR}. Expected subfolders like '0', '1'.")
    latest = str(max(candidates))
    if model_num:
        latest = model_num
    return os.path.join(SPARSE_DIR, latest)

def check_dense_workspace():
    """Verify undistortion created the expected structure for COLMAP format."""
    required = [
        os.path.join(DENSE_DIR, "images"),   # undistorted images live here
        os.path.join(DENSE_DIR, "sparse"),   # copied sparse model
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Dense workspace invalid; missing: {missing}")

def deep_image_to_dense_connector():
    model_dir = latest_sparse_model_dir()
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(os.path.join(model_dir, "rec.ply"))
    o3d.io.write_point_cloud(os.path.join(DENSE_EE_DIR, "rec.ply"), pcd)
    return
# ----------------------------------
# STEP 1: Sparse Reconstruction
# ----------------------------------
def feature_extraction():
    run_cmd([COLMAP_BIN, "feature_extractor",
             "--database_path", DATABASE_PATH,
             "--image_path", IMAGES_DIR])

def feature_matching():
    run_cmd([COLMAP_BIN, "exhaustive_matcher",
             "--database_path", DATABASE_PATH])

def sparse_mapping():
    run_cmd([COLMAP_BIN, "mapper",
             "--database_path", DATABASE_PATH,
             "--image_path", IMAGES_DIR,
             "--output_path", SPARSE_DIR])

# ----------------------------------
# STEP 2: Dense Reconstruction
# ----------------------------------
def image_undistortion(model_num=None):
    sparse_model = latest_sparse_model_dir(model_num)
    run_cmd([COLMAP_BIN, "image_undistorter",
             "--image_path", IMAGES_DIR,
             "--input_path", sparse_model,
             "--output_path", DENSE_DIR,
             "--output_type", "COLMAP"])  # produces dense/images + dense/sparse + dense/stereo

def dense_stereo_try(gpu_index=0, max_image_size=2400, geom_consistency=True):
    """Run PatchMatchStereo; raise if it fails."""
    args = [COLMAP_BIN, "patch_match_stereo",
            "--workspace_path", DENSE_DIR,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", str(max_image_size)]
    if geom_consistency:
        args += ["--PatchMatchStereo.geom_consistency", "true"]
    # Prefer GPU index if non-negative, else CPU
    if gpu_index is not None:
        args += ["--PatchMatchStereo.gpu_index", str(gpu_index)]
    run_cmd(args)

def dense_stereo_with_fallback():
    """
    Try GPU first; on crash, fall back to CPU and smaller images.
    This is to mitigate exit code 3221226505 (stack buffer overrun) on Windows.
    """
    check_dense_workspace()
    try:
        print("→ Trying PatchMatchStereo on GPU (gpu_index=0)")
        dense_stereo_try(gpu_index=0, max_image_size=2400, geom_consistency=True)
    except subprocess.CalledProcessError as e:
        print("\nPatchMatchStereo GPU run failed.")
        print("Return code:", e.returncode)
        # Common Windows crash: 3221226505 == 0xC0000409 stack buffer overrun
        # Fall back to CPU and lower resolution
        print("→ Falling back to CPU (gpu_index=-1) with reduced max_image_size=2000")
        dense_stereo_try(gpu_index=-1, max_image_size=2000, geom_consistency=True)

def stereo_fusion():
    run_cmd([COLMAP_BIN, "stereo_fusion",
             "--workspace_path", DENSE_DIR,
             "--workspace_format", "COLMAP",
             "--input_type", "geometric",
             "--output_path", os.path.join(DENSE_DIR, "fused.ply")])

# ----------------------------------
# STEP 3: Mesh Generation
# ----------------------------------
def poisson_meshing(dense_dir=DENSE_DIR, input_cloud="fused.ply", ):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(os.path.join(dense_dir,input_cloud))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    o3d.io.write_point_cloud(os.path.join(dense_dir, "fused_fixed.ply"), pcd)

    fused = os.path.join(dense_dir, input_cloud)
    fused = os.path.join(dense_dir, "fused_fixed.ply")
    if not os.path.exists(fused):
        raise FileNotFoundError(f"Missing fused point cloud at {fused}.")
    run_cmd([COLMAP_BIN, "poisson_mesher",
             "--input_path", fused,
             "--output_path", os.path.join(dense_dir, "meshed-poisson.ply"),
             "--PoissonMeshing.depth",  str(13),
             "--PoissonMeshing.trim",  str(5)])

# ----------------------------------
# STEP 4: Texture Mapping ##TODO: This does not work
# ----------------------------------
def texture_mapping():
    mesh_path = os.path.join(DENSE_DIR, "meshed-poisson.ply")
    undistorted_images = os.path.join(DENSE_DIR, "images")  # IMPORTANT: use undistorted images
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Missing mesh for texturing at {mesh_path}.")
    if not os.path.exists(undistorted_images):
        raise FileNotFoundError(f"Missing undistorted images at {undistorted_images}.")
    run_cmd([COLMAP_BIN, "texture_mapper",
             "--input_path", mesh_path,
             "--image_path", undistorted_images,
             "--output_path", os.path.join(DENSE_DIR, "textured.obj")])

# ----------------------------------
# MAIN PIPELINE
# ----------------------------------
if __name__ == "__main__":
    print("Starting COLMAP full pipeline (SfM → MVS → Mesh → Texture)…")
    # feature_extraction()
    # feature_matching()
    # sparse_mapping()
    # image_undistortion()
    # dense_stereo_with_fallback()
    # stereo_fusion()
    # poisson_meshing()
    # dense_ee = os.path.join(WORKSPACE, "results_loftr_matching_lowres_quality_high_0046", "dense_ee")
    deep_image_to_dense_connector()
    poisson_meshing(dense_dir=DENSE_EE_DIR, input_cloud="rec.ply")

    print("\nDone! Check outputs:")
    print(f"- Sparse models: {SPARSE_DIR}\\<id>")
    print(f"- Dense point cloud: {DENSE_DIR}\\fused.ply")
    print(f"- Mesh: {DENSE_DIR}\\meshed-poisson.ply")
    print(f"- Textured mesh: {DENSE_DIR}\\textured.obj")
