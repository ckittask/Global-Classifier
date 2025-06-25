import { datasetsEndpoints } from 'utils/endpoints';
import apiDev from './api-dev';

export async function getDatasetsOverview(
  pageNum: number,
  sort: string
) {
  const { data } = await apiDev.get(datasetsEndpoints.GET_OVERVIEW(), {
    params: {
      page: pageNum,
      generationStatus: "all",
      sortBy:sort?.split(" ")?.[0],
      sortType: sort?.split(" ")?.[1],
      pageSize: 12,
    },
  });
  return data?.response ?? [];
}