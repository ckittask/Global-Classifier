DATA_DOWNLOAD_ENDPOINT = "http://file-handler:8000/datasetgroup/data/download/json"

GET_DATASET_METADATA_ENDPOINT = (
    "http://ruuter-private:8088/classifier/datasetgroup/group/metadata"
)

GET_MODEL_METADATA_ENDPOINT = "http://ruuter-private:8088/classifier/datamodel/metadata"

UPDATE_MODEL_TRAINING_STATUS_ENDPOINT = (
    "http://ruuter-private:8088/classifier/datamodel/update/training/status"
)

CREATE_TRAINING_PROGRESS_SESSION_ENDPOINT = (
    "http://ruuter-private:8088/classifier/datamodel/progress/create"
)

UPDATE_TRAINING_PROGRESS_SESSION_ENDPOINT = (
    "http://ruuter-private:8088/classifier/datamodel/progress/update"
)

TEST_DEPLOYMENT_ENDPOINT = (
    "http://deployment-service:8003/classifier/datamodel/deployment/testing/update"
)

TRAINING_LOGS_PATH = "/app/model_trainer/training_logs.log"

MODEL_RESULTS_PATH = "/shared/model_trainer/results"  # stored in the shared folder which is connected to s3-ferry

LOCAL_BASEMODEL_TRAINED_LAYERS_SAVE_PATH = "/shared/model_trainer/results/{model_id}/trained_base_model_layers"  # stored in the shared folder which is connected to s3-ferry

LOCAL_CLASSIFICATION_LAYER_SAVE_PATH = "/shared/model_trainer/results/{model_id}/classifier_layers"  # stored in the shared folder which is connected to s3-ferry

LOCAL_LABEL_ENCODER_SAVE_PATH = "/shared/model_trainer/results/{model_id}/label_encoders"  # stored in the shared folder which is connected to s3-ferry

S3_FERRY_MODEL_STORAGE_PATH = "/models"  # folder path in s3 bucket

S3_FERRY_ENDPOINT = "http://s3-ferry:3000/v1/files/copy"

BASE_MODEL_FILENAME = "base_model_trainable_layers_{model_id}"

CLASSIFIER_MODEL_FILENAME = "classifier_{model_id}.pth"

MODEL_TRAINING_IN_PROGRESS = "training in-progress"

MODEL_TRAINING_SUCCESSFUL = "trained"

MODEL_TRAINING_FAILED = "not trained"


# MODEL TRAINING PROGRESS SESSION CONSTANTS

INITIATING_TRAINING_PROGRESS_STATUS = "Initiating Training"

TRAINING_IN_PROGRESS_PROGRESS_STATUS = "Training In-Progress"

DEPLOYING_MODEL_PROGRESS_STATUS = "Deploying Model"

MODEL_TRAINED_AND_DEPLOYED_PROGRESS_STATUS = "Model Trained And Deployed"


INITIATING_TRAINING_PROGRESS_MESSAGE = "Download and preparing dataset"

TRAINING_IN_PROGRESS_PROGRESS_MESSAGE = (
    "The dataset is being trained on all selected models"
)

DEPLOYING_MODEL_PROGRESS_MESSAGE = (
    "Model training complete. The trained model is now being deployed"
)

MODEL_TRAINED_AND_DEPLOYED_PROGRESS_MESSAGE = (
    "The model was trained and deployed successfully to the environment"
)

MODEL_TRAINING_FAILED_ERROR = "Training Failed"


INITIATING_TRAINING_PROGRESS_PERCENTAGE = 30

TRAINING_IN_PROGRESS_PROGRESS_PERCENTAGE = 50

DEPLOYING_MODEL_PROGRESS_PERCENTAGE = 80

MODEL_TRAINED_AND_DEPLOYED_PROGRESS_PERCENTAGE = 100


# OOD TRAINING CONSTANTS (Added for enhanced functionality)

# OOD Training Status Messages
OOD_TRAINING_PROGRESS_MESSAGE = (
    "The dataset is being trained with OOD detection capabilities"
)

# OOD Configuration Defaults
DEFAULT_OOD_WEIGHT = 0.1
DEFAULT_ENERGY_MARGIN = 10.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_USE_SPECTRAL_NORM = False
DEFAULT_UNCERTAINTY_METHOD = "entropy"

# OOD Model Storage Paths
OOD_METRICS_SAVE_PATH = "/shared/model_trainer/results/{model_id}/ood_metrics"
OOD_CONFIG_SAVE_PATH = "/shared/model_trainer/results/{model_id}/ood_config.json"

# Supported Uncertainty Methods
SUPPORTED_UNCERTAINTY_METHODS = ["entropy", "energy", "combined"]

# OOD Training Types
OOD_TRAINING_ENABLED = "ood_enabled"
OOD_TRAINING_DISABLED = "ood_disabled"

# Supported Models for Testing
SUPPORTED_MODELS = {
    "estbert": "tartuNLP/EstBERT",
    "estbert-small": "tartuNLP/EstBERT-small",
    "xlm-roberta": "xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "multilingual-bert": "bert-base-multilingual-cased",
}
