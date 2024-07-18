python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" --config "3d_fullres" --fold 0 --gpu_id 3 --trainer_class_name "nnUNetTrainer_1epoch" --export_validation_probabilities False


python -m monai.apps.nnunet nnUNetV2Runner predict --input_config "./input.yaml" --list_of_lists_or_source_folder "/mnt/Data01/nnUNet/dataset/nnUNet_raw/Dataset559_data00_cls6/imagesTs_100/" --output_folder "/mnt/Data01/nnUNet/output/nii/Dataset559_data00_cls6/monai" --model_training_output_dir "/mnt/Data02/project/nnUNet/dataset/nnUNet_results/Dataset559_data00_cls6/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_from_nnunetcode/" --use_folds "0, " --tile_step_size 0.5 --use_gaussian True --use_mirroring False --checkpoint_name 'checkpoint_best.pth' --num_processes_preprocessing 1 --num_processes_segmentation_export 1 --overwrite False



