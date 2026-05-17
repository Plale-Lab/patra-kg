-- Follow-up to 202605_drop_workflow_and_revamp_ckn.sql
-- Drop the orphan review_submission_id column on generated_dataset_artifacts.
-- The submission_queue table it referenced was dropped in the prior migration;
-- this column has no code references anywhere in the repo.
ALTER TABLE generated_dataset_artifacts DROP COLUMN IF EXISTS review_submission_id;
