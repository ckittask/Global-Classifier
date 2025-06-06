
-- changeset erangi:global-classifier-centops-ckb-tables
CREATE TABLE public.mock_centops (
    id          SERIAL PRIMARY KEY,
    agency_id   VARCHAR(255) NOT NULL,
    agency_name VARCHAR(255) NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE public.mock_ckb (
    agency_id         VARCHAR(50) PRIMARY KEY,
    agency_data_hash  VARCHAR(255) NOT NULL,
    data_url          TEXT NOT NULL,
    created_at        TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO public.mock_centops (agency_id, agency_name, created_at) VALUES
    ('1', 'ID Department', NOW()),
    ('2', 'Tax Department', NOW());

INSERT INTO public.mock_ckb (agency_id, agency_data_hash, data_url, created_at) VALUES
    ('1', 'hash_dummy_1', 'https://example.com/signed-url-1', NOW()),
    ('2', 'hash_dummy_2', 'https://example.com/signed-url-2', NOW());

