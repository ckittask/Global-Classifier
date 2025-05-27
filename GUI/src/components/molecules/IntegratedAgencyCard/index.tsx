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

type IntegratedAgencyCardProps = {
  agencyId: string;
  agencyName?: string;
  lastTrained?: Date | null |string;
  isLatest?: boolean;
  isEnabled?: boolean;
  lastUpdated?: Date | null;
  lastUsed?: Date | null;
  validationStatus?: string;
  syncStatus?: string;
  lastModelTrained?: string;
  lastSynced?: Date | null;
  enableAllowed?: boolean;	
};

const IntegratedAgencyCard: FC<PropsWithChildren<IntegratedAgencyCardProps>> = ({
  agencyId,
  agencyName,
  isLatest,
  isEnabled,
  lastUpdated,
  lastTrained,
  syncStatus,
  lastModelTrained,
  lastSynced,
  enableAllowed
}) => {

  const { t } = useTranslation();


  return (
    <div>
      <div className="dataset-group-card">
        <div className="row switch-row">
          <div className="text">{agencyName}</div>
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
        <SyncStatusLabel status={syncStatus} />
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
            {t('integratedAgencies.agencyCard.resync')}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default IntegratedAgencyCard;
