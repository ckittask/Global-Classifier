-- Liquibase changeset for creating the datasets table
-- changeset erangiar:datasets-table
CREATE TABLE public.datasets (
    id BIGSERIAL PRIMARY KEY,
    major INTEGER NOT NULL,
    minor INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    generation_status VARCHAR(64) NOT NULL,
    last_model_trained VARCHAR(255),
    last_trained TIMESTAMP WITH TIME ZONE
);