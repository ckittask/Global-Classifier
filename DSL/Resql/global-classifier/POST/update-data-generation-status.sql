UPDATE public.datasets
SET generation_status = :generationStatus
WHERE id = :datasetId::bigint;