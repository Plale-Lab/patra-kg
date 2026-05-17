-- Read-only schema + data-shape inspection for production patradb.
-- Single SELECT — produces ONE result table you can export from DBeaver.
--
-- DBeaver:  paste into SQL editor → Execute (Ctrl+Enter / Cmd+Enter) →
--           right-click result table → Export Data → CSV → send the file back.
--
-- All values are text for uniformity; `section` column lets you split it later.
-- No row content is exposed — only metadata + counts.
--
-- The data-shape queries (sections 08+) use query_to_xml() so they don't fail
-- if a column happens to be missing on the target DB — the row just shows
-- "column_not_present".

WITH

tables AS (
  SELECT
    '01_tables'::text AS section,
    table_name        AS key,
    ''::text          AS value
  FROM information_schema.tables
  WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
),

columns_audit AS (
  SELECT
    '02_columns'::text                                                                   AS section,
    table_name || '.' || column_name                                                     AS key,
    data_type
      || (CASE WHEN is_nullable = 'YES' THEN ' NULL' ELSE ' NOT NULL' END)
      || COALESCE(' DEFAULT ' || column_default, '')                                     AS value
  FROM information_schema.columns
  WHERE table_schema = 'public'
    AND table_name IN (
      'users', 'edge_devices', 'model_cards', 'datasheets', 'models',
      'experiments', 'raw_images', 'experiment_images', 'events', 'power_summary'
    )
),

constraints_audit AS (
  SELECT
    '03_constraints'::text                                          AS section,
    tc.table_name || '.' || tc.constraint_name                      AS key,
    tc.constraint_type
      || ' (' || string_agg(kcu.column_name, ',' ORDER BY kcu.ordinal_position) || ')'
      || COALESCE(
           ' -> ' || (
             SELECT ccu.table_name || '(' || string_agg(ccu.column_name, ',') || ')'
             FROM information_schema.constraint_column_usage ccu
             WHERE ccu.constraint_name = tc.constraint_name
               AND tc.constraint_type = 'FOREIGN KEY'
             GROUP BY ccu.table_name
           ),
           ''
         )                                                          AS value
  FROM information_schema.table_constraints tc
  LEFT JOIN information_schema.key_column_usage kcu
    ON kcu.constraint_name = tc.constraint_name
   AND kcu.table_schema   = tc.table_schema
  WHERE tc.table_schema = 'public'
    AND tc.table_name IN (
      'users', 'edge_devices', 'model_cards', 'datasheets', 'models',
      'experiments', 'raw_images', 'experiment_images', 'events', 'power_summary'
    )
  GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
),

indexes_audit AS (
  SELECT
    '04_indexes'::text             AS section,
    tablename || '.' || indexname  AS key,
    indexdef                       AS value
  FROM pg_indexes
  WHERE schemaname = 'public'
    AND tablename IN (
      'users', 'edge_devices', 'model_cards', 'datasheets', 'models',
      'experiments', 'raw_images', 'experiment_images', 'events', 'power_summary'
    )
),

triggers_audit AS (
  SELECT
    '05_triggers'::text                                                 AS section,
    event_object_table || '.' || trigger_name                           AS key,
    event_manipulation || ' ' || action_timing || ' ' || action_statement AS value
  FROM information_schema.triggers
  WHERE trigger_schema = 'public'
),

trigger_fns AS (
  SELECT
    '06_trigger_functions'::text                          AS section,
    proname                                               AS key,
    LEFT(pg_get_functiondef(p.oid), 400)                  AS value
  FROM pg_proc p
  JOIN pg_namespace n ON p.pronamespace = n.oid
  WHERE n.nspname = 'public'
    AND proname IN ('fn_ingest_camera_trap_event', 'fn_ingest_camera_trap_power')
),

row_counts AS (
  SELECT '07_row_counts' AS section, 'users'             AS key, COUNT(*)::text AS value FROM users
  UNION ALL SELECT '07_row_counts', 'edge_devices',          COUNT(*)::text FROM edge_devices
  UNION ALL SELECT '07_row_counts', 'model_cards',           COUNT(*)::text FROM model_cards
  UNION ALL SELECT '07_row_counts', 'datasheets',            COUNT(*)::text FROM datasheets
  UNION ALL SELECT '07_row_counts', 'models',                COUNT(*)::text FROM models
  UNION ALL SELECT '07_row_counts', 'experiments',           COUNT(*)::text FROM experiments
  UNION ALL SELECT '07_row_counts', 'raw_images',            COUNT(*)::text FROM raw_images
  UNION ALL SELECT '07_row_counts', 'experiment_images',     COUNT(*)::text FROM experiment_images
  UNION ALL SELECT '07_row_counts', 'events',                COUNT(*)::text FROM events
  UNION ALL SELECT '07_row_counts', 'power_summary',         COUNT(*)::text FROM power_summary
),

-- Helper: defer-parsing scalar SQL evaluator. query_to_xml parses the inner
-- SQL only at runtime, so it's safe to query columns that may or may not exist.
-- Returns the text value of the first column of the first row (or NULL).
-- Used for sections 08+ where the column might be absent.

users_shape AS (
  SELECT '08_users_shape'::text AS section, 'total_rows'::text AS key, COUNT(*)::text AS value FROM users
  UNION ALL SELECT '08_users_shape', 'username_present',
         (EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema='public' AND table_name='users' AND column_name='username'))::text
  UNION ALL SELECT '08_users_shape', 'username_null_or_empty',
         CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema='public' AND table_name='users' AND column_name='username')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(*) AS count FROM public.users WHERE username IS NULL OR username = ''$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'column_not_present' END
  UNION ALL SELECT '08_users_shape', 'ckn_user_id_present',
         (EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema='public' AND table_name='users' AND column_name='ckn_user_id'))::text
  UNION ALL SELECT '08_users_shape', 'ckn_user_id_null_or_empty',
         CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema='public' AND table_name='users' AND column_name='ckn_user_id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(*) AS count FROM public.users WHERE ckn_user_id IS NULL OR ckn_user_id = ''$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'column_not_present' END
  UNION ALL SELECT '08_users_shape', 'effective_username_distinct',
         CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema='public' AND table_name='users' AND column_name='ckn_user_id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(DISTINCT COALESCE(username, ckn_user_id)) AS count FROM public.users$q$,
                      true, true, ''
                    )))[1]::text
              ELSE (SELECT COUNT(DISTINCT username)::text FROM users) END
),

devices_shape AS (
  SELECT '09_devices_shape'::text AS section, 'total_rows' AS key, COUNT(*)::text AS value FROM edge_devices
  UNION ALL SELECT '09_devices_shape', 'device_id_present',
         (EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema='public' AND table_name='edge_devices' AND column_name='device_id'))::text
  UNION ALL SELECT '09_devices_shape', 'device_id_null_or_empty',
         CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema='public' AND table_name='edge_devices' AND column_name='device_id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(*) AS count FROM public.edge_devices WHERE device_id IS NULL OR device_id = ''$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'column_not_present' END
  UNION ALL SELECT '09_devices_shape', 'device_id_distinct',
         CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema='public' AND table_name='edge_devices' AND column_name='device_id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(DISTINCT device_id) AS count FROM public.edge_devices$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'column_not_present' END
),

experiments_orphans AS (
  SELECT '10_exp_orphans'::text AS section, 'user_id_data_type' AS key,
         (SELECT data_type FROM information_schema.columns
          WHERE table_schema='public' AND table_name='experiments' AND column_name='user_id')::text AS value
  UNION ALL SELECT '10_exp_orphans', 'edge_device_id_data_type',
         (SELECT data_type FROM information_schema.columns
          WHERE table_schema='public' AND table_name='experiments' AND column_name='edge_device_id')
  UNION ALL SELECT '10_exp_orphans', 'experiments_with_user_id_orphan_to_users.id',
         CASE WHEN (SELECT data_type FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='experiments' AND column_name='user_id') = 'bigint'
                  AND EXISTS (SELECT 1 FROM information_schema.columns
                              WHERE table_schema='public' AND table_name='users' AND column_name='id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(*) AS count FROM public.experiments e LEFT JOIN public.users u ON u.id = e.user_id WHERE u.id IS NULL$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'skipped_type_mismatch_or_no_users.id' END
  UNION ALL SELECT '10_exp_orphans', 'experiments_with_edge_device_id_orphan_to_edge_devices.id',
         CASE WHEN (SELECT data_type FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='experiments' AND column_name='edge_device_id') = 'bigint'
                  AND EXISTS (SELECT 1 FROM information_schema.columns
                              WHERE table_schema='public' AND table_name='edge_devices' AND column_name='id')
              THEN (xpath('//row/count/text()',
                    query_to_xml(
                      $q$SELECT COUNT(*) AS count FROM public.experiments e LEFT JOIN public.edge_devices d ON d.id = e.edge_device_id WHERE d.id IS NULL$q$,
                      true, true, ''
                    )))[1]::text
              ELSE 'skipped_type_mismatch_or_no_edge_devices.id' END
  UNION ALL SELECT '10_exp_orphans', 'experiments_with_model_id_orphan_to_models.id',
         (xpath('//row/count/text()',
           query_to_xml(
             $q$SELECT COUNT(*) AS count FROM public.experiments e LEFT JOIN public.models m ON m.id = e.model_id WHERE m.id IS NULL$q$,
             true, true, ''
           )))[1]::text
),

experiment_images_orphans AS (
  SELECT '11_eimg_orphans'::text AS section,
         'experiment_images.experiment_id_orphan_to_experiments.id' AS key,
         (xpath('//row/count/text()',
           query_to_xml(
             $q$SELECT COUNT(*) AS count FROM public.experiment_images ei LEFT JOIN public.experiments e ON e.id = ei.experiment_id WHERE e.id IS NULL$q$,
             true, true, ''
           )))[1]::text AS value
  UNION ALL SELECT '11_eimg_orphans',
         'experiment_images.raw_image_id_orphan_to_raw_images.id',
         (xpath('//row/count/text()',
           query_to_xml(
             $q$SELECT COUNT(*) AS count FROM public.experiment_images ei LEFT JOIN public.raw_images r ON r.id = ei.raw_image_id WHERE r.id IS NULL$q$,
             true, true, ''
           )))[1]::text
),

versioning_state AS (
  SELECT '12_versioning'::text AS section, 'model_cards.asset_version present'      AS key,
         (EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_name='model_cards' AND column_name='asset_version'))::text AS value
  UNION ALL SELECT '12_versioning', 'datasheets.asset_version present',
         (EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_name='datasheets' AND column_name='asset_version'))::text
  UNION ALL SELECT '12_versioning', 'model_cards_uuid_version_key constraint',
         (EXISTS (SELECT 1 FROM pg_constraint
                  WHERE conname='model_cards_uuid_version_key'))::text
  UNION ALL SELECT '12_versioning', 'datasheets_uuid_version_key constraint',
         (EXISTS (SELECT 1 FROM pg_constraint
                  WHERE conname='datasheets_uuid_version_key'))::text
),

uuid_shape AS (
  SELECT '13_uuid_shape'::text AS section, 'model_cards rows with NULL uuid' AS key,
         COUNT(*)::text AS value FROM model_cards WHERE uuid IS NULL
  UNION ALL SELECT '13_uuid_shape', 'model_cards uuid duplicates',
         (SELECT COUNT(*) FROM (
            SELECT uuid FROM model_cards WHERE uuid IS NOT NULL
            GROUP BY uuid HAVING COUNT(*)>1
          ) x)::text
  UNION ALL SELECT '13_uuid_shape', 'datasheets rows with NULL uuid',
         COUNT(*)::text FROM datasheets WHERE uuid IS NULL
  UNION ALL SELECT '13_uuid_shape', 'datasheets uuid duplicates',
         (SELECT COUNT(*) FROM (
            SELECT uuid FROM datasheets WHERE uuid IS NOT NULL
            GROUP BY uuid HAVING COUNT(*)>1
          ) x)::text
),

extras AS (
  SELECT '14_meta'::text AS section, 'postgres_version' AS key, version() AS value
  UNION ALL SELECT '14_meta', 'extension:' || extname, extversion FROM pg_extension
)

SELECT section, key, value FROM tables
UNION ALL SELECT section, key, value FROM columns_audit
UNION ALL SELECT section, key, value FROM constraints_audit
UNION ALL SELECT section, key, value FROM indexes_audit
UNION ALL SELECT section, key, value FROM triggers_audit
UNION ALL SELECT section, key, value FROM trigger_fns
UNION ALL SELECT section, key, value FROM row_counts
UNION ALL SELECT section, key, value FROM users_shape
UNION ALL SELECT section, key, value FROM devices_shape
UNION ALL SELECT section, key, value FROM experiments_orphans
UNION ALL SELECT section, key, value FROM experiment_images_orphans
UNION ALL SELECT section, key, value FROM versioning_state
UNION ALL SELECT section, key, value FROM uuid_shape
UNION ALL SELECT section, key, value FROM extras
ORDER BY section, key;
