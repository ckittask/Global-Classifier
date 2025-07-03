import { FC } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FormCheckboxes,
  FormInput,
  FormRadios,
  FormSelect,
  Label,
} from 'components';
import { formattedArray, toLabelValueArray } from 'utils/commonUtilts';
import { useQuery } from '@tanstack/react-query';
import CircularSpinner from '../CircularSpinner/CircularSpinner';
import { DataModel } from 'types/dataModels';
import { dataModelsQueryKeys, datasetQueryKeys } from 'utils/queryKeys';
import { getDeploymentEnvironments } from 'services/datamodels';
import { getAllDatasetVersions } from 'services/datasets';

type DataModelFormType = {
  dataModel: any;
  handleChange: (name: keyof DataModel, value: any) => void;
  errors?: Record<string, string>;
  type: string;
};

const DataModelForm: FC<DataModelFormType> = ({
  dataModel,
  handleChange,
  errors,
  type,
}) => {
  const { t } = useTranslation();
   const { data: deploymentEnvironmentsData } = useQuery({
    queryKey: datasetQueryKeys.DATASET_VERSIONS(),
    queryFn: () => getDeploymentEnvironments(),
  });

  const { data: datasetVersions } = useQuery({
    queryKey: dataModelsQueryKeys.DATA_MODEL_DEPLOYMENT_ENVIRONMENTS(),
    queryFn: () => getAllDatasetVersions(),
  });
  
  return (
    <div>
      {type === 'create' ? (
        <div>
          <div className="grey-card">
            <FormInput
              name="modelName"
              label="Model Name"
              value={dataModel.modelName}
              onChange={(e) => handleChange('modelName', e.target.value)}
              error={errors?.modelName}
            />
          </div>
          <div className="grey-card">
            {t('dataModels.dataModelForm.modelVersion')}{' '}
            <Label type="success">{dataModel?.version}</Label>
          </div>
        </div>
      ) : (
        <div className="grey-card flex-grid">
          <div className="title">{dataModel.modelName}</div>
          <Label type="success">{dataModel?.version}</Label>
        </div>
      )}

      {((type === 'configure') || type === 'create')
         ? (
        <div>
          <div className="title-sm">
            {t('dataModels.dataModelForm.datasetGroup')}{' '}
          </div>
          <div className="grey-card" style={{
            display: "flex",
            flexDirection: "column"
          }} >
            <FormSelect
              name="datasetId"
              options={toLabelValueArray(datasetVersions, 'id','version')??[]}
              label=""
              onSelectionChange={(selection) => {
                handleChange('datasetId', selection?.value);
              }}
              value={dataModel?.datasetId === null && t('dataModels.dataModelForm.errors.datasetVersionNotExist')}
              defaultValue={dataModel?.datasetId ? dataModel?.datasetId : t('dataModels.dataModelForm.errors.datasetVersionNotExist')}
              error={errors?.datasetId}
            />
            <div>
              {(type === 'configure') && !dataModel.datasetId && <span style={{
                color: "red", fontSize: "13px"
              }}>{t('dataModels.dataModelForm.errors.datasetVersionNotExist')}</span>}
            </div>
          </div>

          <div className="title-sm">
            {t('dataModels.dataModelForm.baseModels')}{' '}
          </div>
          <div className="grey-card flex-grid">
            <FormCheckboxes
              isStack={false}
              items={formattedArray(deploymentEnvironmentsData?.[0]?.baseModels)??[]}
              name="baseModels"
              label=""
              onValuesChange={(values) =>
                handleChange('baseModels', values.baseModels)
              }
              error={errors?.baseModels}
              selectedValues={dataModel?.baseModels}
            />
          </div>

          <div className="title-sm">
            {t('dataModels.dataModelForm.deploymentPlatform')}{' '}
          </div>
          <div className="grey-card">
            <FormRadios
              items={formattedArray(deploymentEnvironmentsData?.[0]?.deploymentEnvironments)??[]}
              label=""
              name="deploymentEnvironment"
              onChange={(value) => handleChange('deploymentEnvironment', value)}
              error={errors?.deploymentEnvironment}
              selectedValue={dataModel?.deploymentEnvironment}
            />
          </div>
        </div>
      ) : (
        <CircularSpinner />
      )}
    </div>
  );
};

export default DataModelForm;
