# =====================================================================================
# Radiomics Feature Extraction for 100 Brain ROIs (Regions of Interest), Revised version
# =====================================================================================
# This code extracts a comprehensive set of radiomics features from medical images for 
# 100 brain ROIs. Radiomics is a field that utilizes data-characterization algorithms 
# to extract a large number of features from medical images. These features help in 
# quantifying tumor phenotypes and are essential for predictive modeling in clinical 
# outcomes.

# Usage Instructions:
# 1. Locate the "path" variable:
#    - Change the "path" variable to the folder location where your extracted GM (Grey Matter),
#      WM (White Matter), and croppedT1 images are stored. These images should be generated 
#      using c3d (Convert3D), a command-line tool for converting and manipulating 3D medical images.

# 2. Adjust the path for both GM and WM sections:
#    - Repeat the above step for both the grey matter and white matter sections of the code 
#      to ensure both parts are pointing to the correct directories.

# 3. Locate the "output_csv" variable:
#    - Change the "output_csv" variable to the desired location where you want the output CSV 
#      file to be saved. This CSV file will contain the extracted radiomic features.

# 4. Adjust the output location for both GM and WM sections:
#    - Repeat the above step for both the grey matter and white matter sections of the code 
#      to ensure both parts output their results to the correct locations.

# Input Details:
# 1) Cropped T1 image (extension: _croppedT1):
#    - This is the cropped version of the original MRI T1-weighted image, trimmed to match 
#      the size of the grey and white matter regions. These images serve as the reference 
#      for locating and analyzing the corresponding ROIs.

# 2) Grey matter image (extension: label_):
#    - These images contain the extracted grey matter ROIs for 100 brain regions. Grey matter 
#      is the darker tissue of the brain and spinal cord, consisting mainly of nerve cell bodies 
#      and branching dendrites.

# 3) White matter image (extension: labelWM_):
#    - These images contain the extracted white matter ROIs. White matter is the paler tissue 
#      of the brain and spinal cord, consisting mainly of nerve fibers with their myelin sheaths. 
#      The ROIs are defined as being 2mm adjacent to the corresponding white matter regions for 
#      the 100 brain regions.

# By following these steps and properly setting the variables, this code will process the specified 
# MRI images and output a comprehensive set of radiomic features for further analysis.

# Author: Ademola Ilesanmi(PhD)
# Date: 01-July-2024
# =====================================================================================

from radiomics import setVerbosity, featureextractor
import os
import SimpleITK as sitk
import pandas as pd

# Set verbosity level
setVerbosity(60)

# List of x values to process
x_values = [170, 171]

# Initialize the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Initialize a log for empty masks
empty_masks_log = []

# Base path for the data
base_path = '/Users/demoranky/documents/check_all_ABC'
output_base_path = '/Users/demoranky/documents/GLCM_ABC'

# Function to process images and extract features
def process_images(label_type, x_values, output_suffix):
    for x in x_values:
        all_features = []
        print(f"Processing {label_type} images for x={x} in: {base_path}")

        # Loop through files in the directory
        for filename in os.listdir(base_path):
            if filename.endswith(f"_croppedT1_{x}_.nii.gz"):
                image_path = os.path.join(base_path, filename)
                image = sitk.ReadImage(image_path)

                # Find the corresponding mask file
                mask_filename = filename.replace(f"_croppedT1_{x}_", f"{label_type}_{x}_")
                mask_path = os.path.join(base_path, mask_filename)

                if os.path.exists(mask_path):
                    mask = sitk.ReadImage(mask_path)

                    # Check if the mask is empty
                    if not sitk.GetArrayFromImage(mask).any():
                        print(f"Empty mask for file: {mask_filename}, skipping...")
                        empty_masks_log.append(mask_path)
                        continue

                    # Extract features using PyRadiomics
                    feature_vector = extractor.execute(image, mask, label=1)
                    features_df = pd.DataFrame([feature_vector])

                    # Drop unwanted columns
                    columns_to_drop = ["diagnostics_Mask-original_CenterOfMassIndex",
                                       "diagnostics_Mask-original_CenterOfMass"]
                    features_df.drop(columns=columns_to_drop, inplace=True)

                    # Add image and mask paths to the DataFrame
                    features_df['ImageRoot'] = image_path
                    features_df['MaskPath'] = mask_path
                    features_df['Label'] = 1

                    all_features.append(features_df)

        # Check if there are features to concatenate and save
        if all_features:
            # Concatenate all DataFrames for the current x value
            final_df = pd.concat(all_features, ignore_index=True)

            # Save the final DataFrame to a CSV file
            output_csv = os.path.join(output_base_path, f"GLCM_{x}_{output_suffix}.csv")
            final_df.to_csv(output_csv, index=False)

            print(f"Radiomics features for x={x} {label_type} extracted and saved to {output_csv}")
        else:
            print(f"No features found for x={x} {label_type}")

# Process Grey Matter images
process_images("label", x_values, "GM")

# Process White Matter images
process_images("_labelWM", x_values, "WM")

# Save the log of empty masks
empty_masks_log_path = os.path.join(output_base_path, "empty_masks_log.txt")
with open(empty_masks_log_path, 'w') as log_file:
    for mask_path in empty_masks_log:
        log_file.write(f"{mask_path}\n")

print(f"Empty masks log saved to {empty_masks_log_path}")


