import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from fmri_encoder.logger import console
from fmri_encoder.lazy import default_process_multipleX_and_cv_encode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a model.")
    parser.add_argument("--model_names", type=str)
    parser.add_argument("--offsets", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--return_preds", action="store_true", default=False)

    args = parser.parse_args()

    ROOT = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
    tr = 2

    # fMRI data
    data_folder = "english"
    Y = sorted(
        glob.glob(
            os.path.join(
                ROOT, f"data/fMRI/{data_folder}/sub-{args.subject}/func/fMRI*nii.gz"
            )
        )
    )
    masker_path = os.path.join(
        ROOT, "derivatives/fMRI/ROI_masks", f"global_masker_english"
    )

    Xs = []
    offsets = []
    console.log("Loading data...")
    for model_name, offset in zip(args.model_names.split(","), args.offsets.split(",")):
        X_ = sorted(
            glob.glob(
                os.path.join(
                    ROOT,
                    f"data/stimuli-representations/english/{model_name}/activations*",
                )
            )
        )
        X_ = [pd.read_csv(p) for p in X_]
        if ("gpt" in model_name.lower()) and (args.layer is not None):
            cols = [p[[col for col in p.columns if args.layer in col]] for p in X_]
            X_ = [p for p in X_]
        X_ = [p.values for p in X_]
        Xs.append(X_)

        offsets_ = sorted(
            glob.glob(os.path.join(ROOT, f"data/onsets-offsets/english/{offset}_run*"))
        )
        offsets_ = [pd.read_csv(p)["offsets"].values for p in offsets_]
        offsets.append(offsets_)

    output_dict = default_process_multipleX_and_cv_encode(
        Xs, Y, offsets, tr, return_preds=args.return_preds, masker_path=masker_path
    )
    masker = output_dict["masker"]
    folder = os.path.join(ROOT, "derivatives/fMRI/maps/irnlm", f"maps_{data_folder}")
    name = f"sub-{args.subject}_{'-'.join(args.model_names.split(','))}"
    console.log(f"Saving {name} in {folder}")
    # np.save(os.path.join(folder, f"{name}_all_scores.npy"), output_dict['scores'])
    np.save(os.path.join(folder, f"{name}_cv_score.npy"), output_dict["cv_score"])
    nib.save(
        masker.inverse_transform(output_dict["cv_score"]),
        os.path.join(folder, f"{name}_cv_score.nii.gz"),
    )
    if args.return_preds:
        predictions = np.stack(output_dict["predictions"], axis=0)
        nib.save(
            masker.inverse_transform(predictions),
            os.path.join(folder, f"{name}_predictions.nii.gz"),
        )
