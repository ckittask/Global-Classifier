import { FC, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from 'components';
import { Link, useNavigate } from 'react-router-dom';
import './DataModels.scss';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useDialog } from 'hooks/useDialog';
import BackArrowButton from 'assets/BackArrowButton';
import DataModelForm from 'components/molecules/DataModelForm';
import { ButtonAppearanceTypes } from 'enums/commonEnums';
import {
  CreateDataModelPayload,
  DataModel,
  ErrorsType,
} from 'types/dataModels';
import { da } from 'date-fns/locale';

const CreateDataModel: FC = () => {
  const { t } = useTranslation();
  const { open, close } = useDialog();
  const navigate = useNavigate();
  const [availableProdModels, setAvailableProdModels] = useState<string[]>([]);

  const [dataModel, setDataModel] = useState<Partial<DataModel>>({
    modelName: '',
    datasetId: 0,
    baseModels: [],
    deploymentEnvironment: '',
    version: 'V1.0',
  });

  const handleDataModelAttributesChange = (name: string, value: string) => {
    setDataModel((prevFilters) => ({
      ...prevFilters,
      [name]: value,
    }));

    setErrors((prevErrors) => {
      const updatedErrors = { ...prevErrors };

      if (name === 'modelName' && value !== '') {
        delete updatedErrors.modelName;
      }
      if (name === 'baseModels' && value !== '') {
        delete updatedErrors.baseModels;
      }
      if (name === 'deploymentEnvironment' && value !== '') {
        delete updatedErrors.deploymentEnvironment;
      }
      if (name === 'datasetId') {
        delete updatedErrors.datasetId;
      }

      return updatedErrors;
    });
  };

  const [errors, setErrors] = useState<ErrorsType>({
    modelName: '',
    datasetId: '',
    baseModels: '',
    deploymentEnvironment: '',
  });

  const handleCreate = () => {
   console.log(dataModel);
   
  };
  
const isCreateDisabled = () => {
  return (
    !dataModel.modelName ||
    !dataModel.datasetId ||
    !dataModel.baseModels ||
    (Array.isArray(dataModel.baseModels) && dataModel.baseModels.length === 0) ||
    !dataModel.deploymentEnvironment
  );
};

  return (
    <div>
      <div className="container">
        <div className="title_container">
          <div className="flex-grid">
            <Link to={'/data-models'}>
              <BackArrowButton />
            </Link>
            <div className="title">{t('dataModels.createDataModel.title')}</div>
          </div>
        </div>
        <DataModelForm
          errors={errors}
          dataModel={dataModel}
          handleChange={handleDataModelAttributesChange}
          type="create"
        />
      </div>
      <div className="flex data-model-buttons">
        <Button onClick={() => handleCreate()} disabled={isCreateDisabled()} appearance={ButtonAppearanceTypes.PRIMARY}>
          {t('dataModels.createDataModel.title')}
        </Button>
        <Button appearance="secondary" onClick={() => navigate('/data-models')}>
          {t('global.cancel')}
        </Button>
      </div>
    </div>
  );
};

export default CreateDataModel;
