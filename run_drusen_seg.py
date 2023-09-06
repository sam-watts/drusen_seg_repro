import papermill as pm
import sys

MODEL_WEIGHTS = dict(
    image_level = "drusen_segmentation_weights/net_040.pth",
    patch_level = "drusen_segmentation_weights/net_025.pth",
)


# TODO actually implement this switching
DATASETS = dict(
    ODIR = '/data/ODIR_inference',
    STARE = '/data/STARE_inference'
)

DIRS = {
    "baseline": {
        "image_level": dict(
            PATH_DATA = '/data/ODIR_inference',
            PATH_OUTPUT = '/tmp/intermediate_outputs',
        ),
        "patch_level": dict(
            INPUT_DIR_FULL_IMAGE = '/data/ODIR_inference',
            INPUT_DIR_MASK_PRED_FULL_IMAGE = '/tmp/intermediate_outputs',
            OUTPUT_DIR_MASK_PRED_FULL_IMAGE = '/outputs/baseline/odir',
        )
                
    },
    "inpainting": dict(
        
    ),
}

def main():
    mode = sys.argv[1]

    if mode not in ["baseline", "inpainting"]:
        raise ValueError("argument must be one of: [baseline, inpainting]")
        
    run_inference_notebooks(DIRS[mode])
        
    
def run_inference_notebooks(data_dirs: dict, datasets: dict = None):

    # run whole image model
    
    pm.execute_notebook(
        "drusen_seg/test_segmentation_Deep_lab.ipynb",
        "/tmp/test_segmentation_Deep_lab.ipynb",
        parameters=dict(**data_dirs["image_level"], WEIGHTS_PATH=MODEL_WEIGHTS["image_level"]),
        kernel="python",
        log_output=True,
    )

    # run patches model
    pm.execute_notebook(
        "drusen_seg/test_main_model.ipynb",
        "/tmp/test_main_model.ipynb",
        parameters=dict(**data_dirs["patch_level"], WEIGHTS_PATH=MODEL_WEIGHTS["patch_level"]),
        kernel="python3"
    )
    
    # TODO once we are iterating, after each loop, remove intermediate outputs
    
    
if __name__ == "__main__":
   main()