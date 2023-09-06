import papermill as pm
from pathlib import Path

MODEL_WEIGHTS = dict(
    image_level = "drusen_segmentation_weights/net_040.pth",
    patch_level = "drusen_segmentation_weights/net_025.pth",
)

DATASETS = dict(
    ODIR = 'data/ODIR_inference/',
    STARE = 'data/STARE_inference/'
)


def generate_output_path(mode: str, dataset: str):
    return f"outputs/{mode}/{dataset}/"

    
def run_inference_notebooks(mode: dict, dataset: str):
    dataset_path = DATASETS[dataset]
    output_path = generate_output_path(mode, dataset)

    # run whole image model
    pm.execute_notebook(
        "test_segmentation_Deep_lab.ipynb",
        "outputs/test_segmentation_Deep_lab.ipynb",
        parameters=dict(
            WEIGHTS_PATH=MODEL_WEIGHTS["image_level"], 
            PATH_DATA=dataset_path,
            PATH_OUTPUT = 'outputs/intermediate/',
        ),
        kernel="python",
        log_output=True,
    )
    
    # make sure the folder is created for the final output masks
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # run patches model
    pm.execute_notebook(
        "test_main_model.ipynb",
        "outputs/test_main_model.ipynb",
        parameters=dict(
            WEIGHTS_PATH=MODEL_WEIGHTS["patch_level"], 
            INPUT_DIR_FULL_IMAGE=dataset_path,
            INPUT_DIR_MASK_PRED_FULL_IMAGE = 'outputs/intermediate/', 
            OUTPUT_DIR_MASK_PRED_FULL_IMAGE=output_path
        ),
        kernel="python",
        log_output=True,
    )
    
    [f.unlink() for f in Path('outputs/intermediate/').glob("*") if f.is_file()] 
    
def main():
    run_inference_notebooks("baseline", "STARE")
    run_inference_notebooks("baseline", "ODIR")

    
if __name__ == "__main__":
   main()
