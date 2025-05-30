import { FC, PropsWithChildren } from 'react';
import './DatasetGroupCard.scss';
import { Switch } from 'components/FormElements';
import Button from 'components/Button';
import Label from 'components/Label';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useDialog } from 'hooks/useDialog';
import { Operation } from 'types/datasetGroups';
import { datasetQueryKeys } from 'utils/queryKeys';
import { ButtonAppearanceTypes, LabelType } from 'enums/commonEnums';
import { useTranslation } from 'react-i18next';
import { formatDate } from 'utils/commonUtilts';
import SyncStatusLabel from '../SyncStatusLabel';
import DataGenerationStatusLabel from '../DataGenerationStatusLabel';

type DatasetCardProps = {
  datasetId: string;
  datasetName?: string;
  lastTrained?: Date | null |string;
  isLatest?: boolean;
  isEnabled?: boolean;
  lastUpdated?: Date | null;
  dataGenerationStatus?: string;
  lastModelTrained?: string;
  enableAllowed?: boolean;
  majorVersion?: string;
  minorVersion?: string;	
};

const DatasetCard: FC<PropsWithChildren<DatasetCardProps>> = ({
  datasetId,
  datasetName,
  isLatest,
  isEnabled,
  lastUpdated,
  lastTrained,
  dataGenerationStatus,
  lastModelTrained,
  enableAllowed,
  majorVersion,
  minorVersion
}) => {

  const { t } = useTranslation();

  return (
    <div>
      <div className="dataset-group-card">
        <div className="row switch-row">
          <div className="text">{majorVersion && minorVersion ? `${datasetName} Version ${majorVersion}.${minorVersion}`:""}</div>
          <Switch
            label=""
            checked={isEnabled}
            disabled={!enableAllowed}
            onCheckedChange={() => {}}
          />
        </div>
        <div className="py-3">
          <p>
            {t('integratedAgencies.agencyCard.lastModelTrained')}:{' '}
            {lastModelTrained===""?"N/A":lastModelTrained}
          </p>
          <p>
            {t('integratedAgencies.agencyCard.lastUsedForTraining')}:{' '}
            {lastTrained==="1970-01-01T00:00:00.000+00:00"?"N/A":formatDate(lastTrained as Date, 'D.M.yy-H:m')}
          </p>
          <p>
            {t('integratedAgencies.agencyCard.lastSynced')}:{' '}
            {lastSynced && formatDate(lastSynced, 'DD.MM.yy-HH:mm')}
          </p>
        </div>
        

        <div className="flex">
        <DataGenerationStatusLabel status={syncStatus} />
          {isLatest ? (
            <Label type={LabelType.SUCCESS}>
              {t('integratedAgencies.agencyCard.latest')}
            </Label>
          ) : null}
        </div>

        <div className="label-row">
          <Button
            appearance={ButtonAppearanceTypes.SECONDARY}
            size="s"
            onClick={() => {
            }}
          >
            {t('datasetGroups.datasetCard.settings')}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default DatasetCard;
