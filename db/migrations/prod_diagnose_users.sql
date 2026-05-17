-- Diagnostic: show the 16 problem rows so we can decide how to fix them.
-- Read-only, no changes. Run this in DBeaver, export the result.
--
-- We need three pieces of info:
--   A. What's in those 16 rows? (Are they real users or stale test data?)
--   B. Do experiments reference them? (How invasive is fixing/deleting?)
--   C. Quick distribution view (how many of those 16 have at least ckn_user_id?)

-- ============================================================
-- A. The 16 problem rows themselves
-- ============================================================
SELECT id, username, ckn_user_id, created_at, updated_at
FROM users
WHERE username IS NULL OR username = ''
ORDER BY id;

-- ============================================================
-- B. How many experiments reference each of those 16 users?
-- ============================================================
SELECT
  u.id                         AS user_id,
  u.username,
  u.ckn_user_id,
  COUNT(e.id)                  AS experiments_referencing
FROM users u
LEFT JOIN experiments e ON e.user_id = u.id
WHERE u.username IS NULL OR u.username = ''
GROUP BY u.id, u.username, u.ckn_user_id
ORDER BY experiments_referencing DESC, u.id;

-- ============================================================
-- C. Distribution: of the 16, how many have ckn_user_id we could use?
-- ============================================================
SELECT
  CASE
    WHEN ckn_user_id IS NOT NULL AND ckn_user_id <> '' THEN 'has_ckn_user_id (could backfill)'
    WHEN ckn_user_id = ''                              THEN 'ckn_user_id is empty string'
    ELSE                                                    'ckn_user_id is NULL'
  END AS status,
  COUNT(*)
FROM users
WHERE username IS NULL OR username = ''
GROUP BY 1;
