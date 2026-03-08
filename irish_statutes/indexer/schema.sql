-- Irish Statute Book structured storage schema
-- Run once: psql -d vector_db -f indexer/schema.sql

CREATE TABLE IF NOT EXISTS laws (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    year        INTEGER NOT NULL,
    act_number  INTEGER,
    url         TEXT,
    html_path   TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (year, act_number)
);

CREATE TABLE IF NOT EXISTS law_sections (
    id             SERIAL PRIMARY KEY,
    law_id         INTEGER NOT NULL REFERENCES laws(id) ON DELETE CASCADE,
    parent_id      INTEGER REFERENCES law_sections(id) ON DELETE CASCADE,
    section_type   TEXT NOT NULL,
    section_ref    TEXT,
    section_title  TEXT,
    text_content   TEXT,
    position       INTEGER,
    created_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS law_sections_law_id_idx     ON law_sections(law_id);
CREATE INDEX IF NOT EXISTS law_sections_section_ref_idx ON law_sections(section_ref);
CREATE INDEX IF NOT EXISTS law_sections_type_idx        ON law_sections(section_type);
