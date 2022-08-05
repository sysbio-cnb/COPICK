**TRAIN, EVALUATION AND OPTIMIZATION**

# Workflow using the Colony Picking dataset (COPICK) in Detectron2 for training, evaluation and optimization tasks.


TRAINING, EVALUATION AND PANOPTIC PREDICTIONS 

	-"detectron2_train_eval.py": it is used for Detectron2 Config setup to perform model training and evaluation.

	-"detectron2_get_model_predictions.py": it is used to perform model inference on images that can be registered in a dataset or not and save the predictions.

	-"display_colonies_overlay.py": it is used to visualize the predicted colonies overlayed on top of the original image.


HYPERPARAMETERS OPTIMIZATION

	-"detectron2_optimization.py": it is used to perform model hyperparameters optimization in order to choose the best posible values to train and improve the Colony Picking panoptic model in Detectron2. Optimization is done using the library "Optuna" (https://optuna.org/).


