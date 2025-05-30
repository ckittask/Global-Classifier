SELECT 
        mc.agency_id,
        mc.agency_name
    FROM 
        public.mock_centops mc
    WHERE 
        mc.created_at > CASE 
                          WHEN :lastSyncedTimestamp = '' THEN '1970-01-01'::timestamp 
                          ELSE :lastSyncedTimestamp::timestamp 
                        END
    ORDER BY 
        mc.created_at DESC;