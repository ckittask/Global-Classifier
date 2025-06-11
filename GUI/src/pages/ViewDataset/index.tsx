import BackArrowButton from 'assets/BackArrowButton';
import { Button, Card, DataTable, Dialog, Icon, Label, Switch } from 'components';
import { ButtonAppearanceTypes, LabelType } from 'enums/commonEnums';
import React, { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ViewDatasetGroupModalContexts } from 'enums/datasetEnums';
import { generateDynamicColumns } from 'utils/dataTableUtils';
import { MdOutlineDeleteOutline, MdOutlineEdit } from 'react-icons/md';
import { CellContext, ColumnDef, PaginationState } from '@tanstack/react-table';
import {
  DatasetDetails,
  SelectedRowPayload,
} from 'types/datasets';
import SkeletonTable from '../../components/molecules/TableSkeleton/TableSkeleton';
import { sampleDatasetRows } from 'data/sampleDataset';
import DynamicForm from 'components/FormElements/DynamicForm';

const ViewDataset = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 5,
  });
  const [isUpdateModalOpen, setIsUpdateModal] = useState<boolean>(false);
  const [modalAction, setModalAction] = useState<string>("");

  const isMetadataLoading = false; // Placeholder for metadata loading state
  const metadata: any[] = [];
  const isLoading = false; // Placeholder for loading state
  const handleOpenModals = (context: ViewDatasetGroupModalContexts) => {
    if (context === ViewDatasetGroupModalContexts.PATCH_UPDATE_MODAL)
      setIsUpdateModal(true);
    else if (context === ViewDatasetGroupModalContexts.DELETE_ROW_MODAL)
      setModalAction('Delete');
  };
  const datasets = sampleDatasetRows;
  const [updatedDataset, setUpdatedDataset] = useState(datasets?.dataPayload);

  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('datasetId');
  const [selectedRow, setSelectedRow] = useState<SelectedRowPayload>();

  const editView = (props: CellContext<any, unknown>) => {
    return (
      <Button
        appearance={ButtonAppearanceTypes.TEXT}
        onClick={() => {
          setSelectedRow(props.row.original);
          handleOpenModals(ViewDatasetGroupModalContexts.PATCH_UPDATE_MODAL);
        }}
      >
        <Icon icon={<MdOutlineEdit />} />
        {t('global.edit')}
      </Button>
    );
  };

  const deleteView = (props: CellContext<any, unknown>) => (
    <Button
      appearance={ButtonAppearanceTypes.TEXT}
      onClick={() => {
        setSelectedRow(props.row.original);
        handleOpenModals(ViewDatasetGroupModalContexts.DELETE_ROW_MODAL);
      }}
    >
      <Icon icon={<MdOutlineDeleteOutline />} />
      {t('global.delete')}
    </Button>
  );

  const dataColumns = useMemo(
    () => generateDynamicColumns(datasets?.fields ?? [], editView, deleteView),
    [datasets?.fields]
  );

  const patchDataUpdate = (dataRow: SelectedRowPayload) => {
    const payload = updatedDataset?.map((row) =>
      row.rowId === selectedRow?.rowId ? dataRow : row
    );
    setUpdatedDataset(payload);

  };

  return (
    <div className="container">
      <div className="title_container">
        <div className="flex-between">
          <Link to={'/datasets'}>
            <BackArrowButton />
          </Link>
          <div className="title">{t('datasets.detailedView.dataset')} V2.0</div>

        </div>
      </div>
      {isMetadataLoading && <SkeletonTable rowCount={2} />}
      {metadata && !isMetadataLoading && (
        <div>
          <Card
            isHeaderLight={false}
          >
            <div className="flex-between">
              <div>
                <p>
                  {t('datasets.detailedView.version') ?? ''} : V2.0

                </p>
                <p>
                  {t('datasets.detailedView.connectedModels') ?? ''} : 0

                </p>
                <p>
                  {t('datasets.detailedView.noOfItems') ?? ''} : {datasets?.dataPayload?.length ?? 0}
                </p>
              </div>
              <div>
                <Switch label=''></Switch>
                <br />
                <Button appearance='secondary' size='s'>
                  Export Dataset
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
      <div className="mb-20">
        {isLoading && <SkeletonTable rowCount={5} />}
        {!isLoading && updatedDataset && updatedDataset.length > 0 && (
          <DataTable
            data={updatedDataset}
            columns={dataColumns as ColumnDef<string, string>[]}
            pagination={pagination}
            setPagination={(state: PaginationState) => {
              if (
                state.pageIndex === pagination.pageIndex &&
                state.pageSize === pagination.pageSize
              )
                return;
              setPagination(state);
              // getDatasets(state, dgId);
            }}
            pagesCount={datasets?.dataPayload?.length / pagination.pageSize ?? 0}
            isClientSide={false}
          />
        )}
      </div>
      {isUpdateModalOpen && (
        <Dialog
          title={'Edit'}
          onClose={() => setIsUpdateModal(false)}
          isOpen={
            isUpdateModalOpen}
        >
          <DynamicForm
            formData={selectedRow ?? {}}
            onSubmit={patchDataUpdate}
            setPatchUpdateModalOpen={setIsUpdateModal}
          />
        </Dialog>
      )}
    </div>
  );
};

export default ViewDataset;
