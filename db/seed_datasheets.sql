-- Seed 3 real public camera-trap datasets as Patra datasheets.
-- Replaces any prior datasheets in the catalog. Idempotent.
--
-- Datasets:
--   1. Snapshot Serengeti              (Swanson et al., 2015)
--   2. Caltech Camera Traps            (Beery et al., 2018)
--   3. iWildCam 2021 Competition Data  (Beery et al., 2021)
--
-- All metadata fields populated where reasonable: titles, creators, subjects,
-- descriptions, dates, alternate/related identifiers, rights, geo_locations,
-- and funding_references. Publisher rows are reused via ROR/DOI identifiers.

BEGIN;

-- Wipe existing catalog rows (CASCADE → child datasheet_* tables clear too)
TRUNCATE TABLE datasheets RESTART IDENTITY CASCADE;
TRUNCATE TABLE datasheet_publishers RESTART IDENTITY CASCADE;

DO $seed$
DECLARE
  v_now timestamptz := NOW();
  v_pub_dryad   bigint;
  v_pub_caltech bigint;
  v_ds bigint;
BEGIN
  -- ---------------------------------------------------------------------------
  -- Publishers
  -- ---------------------------------------------------------------------------
  INSERT INTO datasheet_publishers (name, publisher_identifier, publisher_identifier_scheme, scheme_uri, lang)
  VALUES ('Dryad Digital Repository', 'https://datadryad.org', 'URL', 'https://datadryad.org/', 'en')
  RETURNING id INTO v_pub_dryad;

  INSERT INTO datasheet_publishers (name, publisher_identifier, publisher_identifier_scheme, scheme_uri, lang)
  VALUES ('California Institute of Technology', 'https://ror.org/05dxps055', 'ROR', 'https://ror.org/', 'en')
  RETURNING id INTO v_pub_caltech;

  -- ===========================================================================
  -- 1. Snapshot Serengeti  (Swanson et al., 2015)
  -- ===========================================================================
  INSERT INTO datasheets (
    publication_year, resource_type, resource_type_general,
    size, format, version, is_private, status,
    created_at, updated_at, publisher_id
  ) VALUES (
    2015, 'Image dataset', 'Dataset',
    '~3.2M images (1.2M sequences, ~9.6 TB)', 'JPEG', '1.0',
    false, 'approved',
    v_now, v_now, v_pub_dryad
  )
  RETURNING identifier INTO v_ds;

  INSERT INTO datasheet_titles (datasheet_id, title, lang) VALUES
    (v_ds, 'Snapshot Serengeti, high-frequency annotated camera trap images of 40 mammalian species in an African savanna', 'en');

  INSERT INTO datasheet_creators (datasheet_id, creator_name, name_type, given_name, family_name, affiliation) VALUES
    (v_ds, 'Swanson, Alexandra', 'Personal', 'Alexandra', 'Swanson', 'University of Minnesota'),
    (v_ds, 'Kosmala, Margaret',  'Personal', 'Margaret',  'Kosmala', 'University of Minnesota'),
    (v_ds, 'Lintott, Chris',     'Personal', 'Chris',     'Lintott', 'University of Oxford'),
    (v_ds, 'Simpson, Robert',    'Personal', 'Robert',    'Simpson', 'Adler Planetarium'),
    (v_ds, 'Smith, Arfon',       'Personal', 'Arfon',     'Smith',   'Adler Planetarium'),
    (v_ds, 'Packer, Craig',      'Personal', 'Craig',     'Packer',  'University of Minnesota');

  INSERT INTO datasheet_subjects (datasheet_id, subject, lang) VALUES
    (v_ds, 'Camera trap',          'en'),
    (v_ds, 'Wildlife monitoring',  'en'),
    (v_ds, 'African mammals',      'en'),
    (v_ds, 'Citizen science',      'en'),
    (v_ds, 'Serengeti ecosystem',  'en');

  INSERT INTO datasheet_descriptions (datasheet_id, description, description_type, lang) VALUES
    (v_ds,
     'Approximately 1.2 million image sequences (~3.2 million individual images) from 225 camera-trap sites deployed across Serengeti National Park, Tanzania, between 2010 and 2013. Identifications of 40 mammalian species were provided by volunteer citizen scientists on the Zooniverse platform.',
     'Abstract', 'en');

  INSERT INTO datasheet_dates (datasheet_id, date, date_type, date_information) VALUES
    (v_ds, '2010-07/2013-05', 'Collected', 'Camera-trap deployment period'),
    (v_ds, '2015-06-09',      'Available', 'Dryad release date');

  INSERT INTO datasheet_alternate_identifiers (datasheet_id, alternate_identifier, alternate_identifier_type) VALUES
    (v_ds, '10.5061/dryad.5pt92', 'DOI');

  INSERT INTO datasheet_related_identifiers (
    datasheet_id, related_identifier, related_identifier_type, relation_type
  ) VALUES
    (v_ds, '10.1038/sdata.2015.26', 'DOI', 'IsDescribedBy');

  INSERT INTO datasheet_rights (datasheet_id, rights, rights_uri, rights_identifier, rights_identifier_scheme, lang) VALUES
    (v_ds, 'Creative Commons Zero v1.0 Universal',
     'https://creativecommons.org/publicdomain/zero/1.0/',
     'CC0-1.0', 'SPDX', 'en');

  INSERT INTO datasheet_geo_locations (datasheet_id, geo_location_place, point_latitude, point_longitude) VALUES
    (v_ds, 'Serengeti National Park, Tanzania', -2.333, 34.833);

  INSERT INTO datasheet_funding_references (
    datasheet_id, funder_name, funder_identifier, funder_identifier_type, scheme_uri, award_number
  ) VALUES (
    v_ds, 'National Science Foundation',
    'https://ror.org/021nxhr62', 'ROR', 'https://ror.org/',
    'DEB-1020558'
  );

  -- ===========================================================================
  -- 2. Caltech Camera Traps  (Beery et al., 2018)
  -- ===========================================================================
  INSERT INTO datasheets (
    publication_year, resource_type, resource_type_general,
    size, format, version, is_private, status,
    created_at, updated_at, publisher_id
  ) VALUES (
    2018, 'Image dataset', 'Dataset',
    '~243 GB (243,100 images)', 'JPEG', '1.0',
    false, 'approved',
    v_now, v_now, v_pub_caltech
  )
  RETURNING identifier INTO v_ds;

  INSERT INTO datasheet_titles (datasheet_id, title, lang) VALUES
    (v_ds, 'Caltech Camera Traps', 'en');

  INSERT INTO datasheet_creators (datasheet_id, creator_name, name_type, given_name, family_name, affiliation) VALUES
    (v_ds, 'Beery, Sara',         'Personal', 'Sara',   'Beery',    'California Institute of Technology'),
    (v_ds, 'Van Horn, Grant',     'Personal', 'Grant',  'Van Horn', 'California Institute of Technology'),
    (v_ds, 'Perona, Pietro',      'Personal', 'Pietro', 'Perona',   'California Institute of Technology');

  INSERT INTO datasheet_subjects (datasheet_id, subject, lang) VALUES
    (v_ds, 'Camera trap',                  'en'),
    (v_ds, 'Computer vision',              'en'),
    (v_ds, 'Out-of-distribution detection', 'en'),
    (v_ds, 'Wildlife',                     'en'),
    (v_ds, 'Machine learning benchmarks',  'en');

  INSERT INTO datasheet_descriptions (datasheet_id, description, description_type, lang) VALUES
    (v_ds,
     'Approximately 243,100 camera-trap images covering 16 mammal species across 140 camera locations in the southwestern United States. Designed as a benchmark for testing generalization of species-recognition models to new locations ("Recognition in Terra Incognita"); locations are split disjoint between train and test.',
     'Abstract', 'en');

  INSERT INTO datasheet_dates (datasheet_id, date, date_type, date_information) VALUES
    (v_ds, '2018', 'Available', 'Initial public release alongside ECCV 2018 paper');

  INSERT INTO datasheet_alternate_identifiers (datasheet_id, alternate_identifier, alternate_identifier_type) VALUES
    (v_ds, 'arXiv:1807.04975', 'arXiv');

  INSERT INTO datasheet_related_identifiers (
    datasheet_id, related_identifier, related_identifier_type, relation_type
  ) VALUES
    (v_ds, '10.1007/978-3-030-01270-0_28', 'DOI', 'IsDescribedBy'),
    (v_ds, 'https://lila.science/datasets/caltech-camera-traps', 'URL', 'IsPartOf');

  INSERT INTO datasheet_rights (datasheet_id, rights, rights_uri, rights_identifier, rights_identifier_scheme, lang) VALUES
    (v_ds, 'Community Data License Agreement – Permissive – Version 1.0',
     'https://cdla.io/permissive-1-0/',
     'CDLA-Permissive-1.0', 'SPDX', 'en');

  INSERT INTO datasheet_geo_locations (datasheet_id, geo_location_place, point_latitude, point_longitude) VALUES
    (v_ds, 'Southwestern United States (Mojave Desert region)', 35.0, -116.5);

  INSERT INTO datasheet_funding_references (
    datasheet_id, funder_name, funder_identifier, funder_identifier_type, scheme_uri, award_title
  ) VALUES (
    v_ds, 'National Science Foundation Graduate Research Fellowship',
    'https://ror.org/021nxhr62', 'ROR', 'https://ror.org/',
    'NSF GRFP'
  );

  -- ===========================================================================
  -- 3. iWildCam 2021 Competition Dataset  (Beery et al., 2021)
  -- ===========================================================================
  INSERT INTO datasheets (
    publication_year, resource_type, resource_type_general,
    size, format, version, is_private, status,
    created_at, updated_at, publisher_id
  ) VALUES (
    2021, 'Image dataset', 'Dataset',
    '~210K training images, ~60K test images', 'JPEG', '2021',
    false, 'approved',
    v_now, v_now, v_pub_caltech
  )
  RETURNING identifier INTO v_ds;

  INSERT INTO datasheet_titles (datasheet_id, title, title_type, lang) VALUES
    (v_ds, 'iWildCam 2021 Competition Dataset', NULL, 'en'),
    (v_ds, 'iWildCam 2021', 'AlternativeTitle', 'en');

  INSERT INTO datasheet_creators (datasheet_id, creator_name, name_type, given_name, family_name, affiliation) VALUES
    (v_ds, 'Beery, Sara',        'Personal', 'Sara',     'Beery',    'California Institute of Technology'),
    (v_ds, 'Agarwal, Arushi',    'Personal', 'Arushi',   'Agarwal',  'California Institute of Technology'),
    (v_ds, 'Cole, Elijah',       'Personal', 'Elijah',   'Cole',     'California Institute of Technology'),
    (v_ds, 'Birodkar, Vighnesh', 'Personal', 'Vighnesh', 'Birodkar', 'Google');

  INSERT INTO datasheet_subjects (datasheet_id, subject, lang) VALUES
    (v_ds, 'Camera trap',                       'en'),
    (v_ds, 'Fine-grained visual categorization', 'en'),
    (v_ds, 'Species recognition',                'en'),
    (v_ds, 'Multi-region wildlife',              'en'),
    (v_ds, 'Domain generalization',              'en');

  INSERT INTO datasheet_descriptions (datasheet_id, description, description_type, lang) VALUES
    (v_ds,
     'A multi-continent camera-trap species-recognition benchmark released for the FGVC8 workshop at CVPR 2021. Combines imagery from camera-trap deployments across multiple regions (the Americas, Africa, and beyond) to test generalization across geography, sensor, and class distribution. Provides per-image species annotations and image-sequence groupings.',
     'Abstract', 'en');

  INSERT INTO datasheet_dates (datasheet_id, date, date_type, date_information) VALUES
    (v_ds, '2021-05-07', 'Available', 'Public release for FGVC8 workshop');

  INSERT INTO datasheet_alternate_identifiers (datasheet_id, alternate_identifier, alternate_identifier_type) VALUES
    (v_ds, 'arXiv:2105.03494', 'arXiv');

  INSERT INTO datasheet_related_identifiers (
    datasheet_id, related_identifier, related_identifier_type, relation_type
  ) VALUES
    (v_ds, 'https://www.kaggle.com/competitions/iwildcam2021-fgvc8', 'URL', 'IsPartOf'),
    (v_ds, 'https://lila.science/datasets/iwildcam-2021-competition-dataset/', 'URL', 'IsPartOf');

  INSERT INTO datasheet_rights (datasheet_id, rights, rights_uri, rights_identifier, rights_identifier_scheme, lang) VALUES
    (v_ds, 'Community Data License Agreement – Permissive – Version 1.0',
     'https://cdla.io/permissive-1-0/',
     'CDLA-Permissive-1.0', 'SPDX', 'en');

  INSERT INTO datasheet_geo_locations (
    datasheet_id, geo_location_place, box_west, box_east, box_south, box_north
  ) VALUES (
    v_ds, 'Multi-region (Americas, Africa, and other camera-trap deployments)',
    -125.0, 50.0, -35.0, 50.0
  );

  INSERT INTO datasheet_funding_references (
    datasheet_id, funder_name, funder_identifier, funder_identifier_type, scheme_uri
  ) VALUES (
    v_ds, 'Google',
    'https://ror.org/006w34k90', 'ROR', 'https://ror.org/'
  );
END
$seed$;

COMMIT;
