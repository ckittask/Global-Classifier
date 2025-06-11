import pandas as pd
import requests
from constants import (
    DATA_DOWNLOAD_ENDPOINT,
    GET_DATASET_METADATA_ENDPOINT,
    TRAINING_LOGS_PATH,
)
from loguru import logger

logger.add(sink=TRAINING_LOGS_PATH)


class DataPipeline:
    def __init__(self, dg_id, cookie):
        logger.info(f"DOWNLOADING DATASET WITH DGID - {dg_id}")

        cookies = {"customJwtCookie": cookie}
        response = requests.get(
            DATA_DOWNLOAD_ENDPOINT, params={"dgId": dg_id}, cookies=cookies
        )

        if response.status_code == 200:
            logger.info("DATA DOWNLOAD SUCCESSFUL")
            data = response.json()
            df = pd.DataFrame(data)
            df = df.drop("rowId", axis=1)
            self.df = df
        else:
            logger.error(
                f"DATA DOWNLOAD FAILED WITH ERROR CODE: {response.status_code}"
            )
            logger.error(f"RESPONSE: {response.text}")
            raise RuntimeError(f"ERROR RESPONSE {response.text}")

        logger.info("****** Calling init function of DataPipeline Class ******")
        logger.info(f"Endpoint : {GET_DATASET_METADATA_ENDPOINT}")

        response_hierarchy = requests.get(
            GET_DATASET_METADATA_ENDPOINT, params={"groupId": dg_id}, cookies=cookies
        )

        if response_hierarchy.status_code == 200:
            logger.info("DATASET HIERARCHY RETRIEVAL SUCCESSFUL")
            hierarchy = response_hierarchy.json()
            self.hierarchy = hierarchy["response"]["data"][0]
        else:
            logger.error(
                f"DATASET HIERARCHY RETRIEVAL FAILED: {response_hierarchy.status_code}"
            )
            logger.error(f"RESPONSE: {response.text}")
            raise RuntimeError(f"ERROR RESPONSE\n {response_hierarchy.text}")

    def extract_input_columns(self):
        """Extract input columns - now simplified to just 'conversation'"""
        validation_rules = self.hierarchy["validationCriteria"]["validationRules"]
        input_columns = [
            key for key, value in validation_rules.items() if not value["isDataClass"]
        ]
        logger.info(f"Input columns: {input_columns}")
        return input_columns

    def extract_target_column(self):
        """Extract the target column for flat classification"""
        validation_rules = self.hierarchy["validationCriteria"]["validationRules"]
        target_columns = [
            key for key, value in validation_rules.items() if value["isDataClass"]
        ]

        if not target_columns:
            raise ValueError("No target column found in validation rules")

        target_column = target_columns[0]
        logger.info(f"Using target column: {target_column}")
        return target_column

    def models_and_filters(self):
        """
        Simplified version for flat classification
        Returns basic model info for compatibility with existing code
        """
        target_column = self.extract_target_column()
        unique_classes = self.df[target_column].unique().tolist()

        models = [{1: unique_classes}]  # Single model with all classes
        filters = [unique_classes]  # Single filter with all classes

        logger.info(f"Flat classification - Classes: {unique_classes}")
        return models, filters

    def create_dataframes(self):
        """Create dataframes for flat classification - SIMPLIFIED"""
        logger.info("CREATING DATAFRAME FOR FLAT CLASSIFICATION")

        input_columns = self.extract_input_columns()
        target_column = self.extract_target_column()

        df = self.df.copy()

        # SIMPLIFIED: Just use the conversation field directly
        if "conversation" in input_columns:
            # Use conversation field as-is
            df["input"] = df["conversation"]
            logger.info("Using 'conversation' field directly as input")
        else:
            # Fallback: combine all input columns (if multiple exist)
            df["input"] = df[input_columns].apply(
                lambda row: " ".join(row.dropna().astype(str)), axis=1
            )
            logger.info(f"Combined input columns: {input_columns}")

        # Rename target column
        df = df.rename(columns={target_column: "target"})

        # Keep only input and target columns
        df = df[["input", "target"]].dropna()

        logger.info(f"Created simplified dataframe with {len(df)} samples")
        logger.info(f"Target classes: {df['target'].unique()}")
        logger.info(f"Class distribution: {df['target'].value_counts().to_dict()}")
        logger.info(f"Sample input: {df['input'].iloc[0][:100]}...")

        return [df]
