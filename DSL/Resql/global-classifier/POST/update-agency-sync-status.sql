UPDATE public.integrated_agencies
    SET 
        sync_status =  :syncStatus::sync_status,
        last_updated_timestamp = CURRENT_TIMESTAMP
    WHERE 
        agency_id = :agencyId;