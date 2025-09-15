-- =============================================================================
-- QUICK SETUP - ESSENTIAL QUERIES FOR YOUR DATABASE
-- Copy and paste these into your Supabase SQL editor
-- =============================================================================

-- 1. CREATE MAIN TABLES
CREATE TABLE IF NOT EXISTS face_embeddings (
    id TEXT NOT NULL DEFAULT ('face_emb_' || substr(gen_random_uuid()::text, 1, 12)) PRIMARY KEY,
    user_detail_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    embedding JSONB NOT NULL,
    embedding_size INTEGER NOT NULL DEFAULT 512,
    confidence REAL NOT NULL DEFAULT 0.0,
    face_quality_score REAL DEFAULT 0.0,
    source_url TEXT,
    enrollment_method TEXT DEFAULT 'api',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT valid_quality CHECK (face_quality_score >= 0.0 AND face_quality_score <= 1.0),
    CONSTRAINT fk_face_embeddings_user_detail FOREIGN KEY (user_detail_id) REFERENCES "UserDetails"(id) ON DELETE CASCADE,
    CONSTRAINT fk_face_embeddings_user FOREIGN KEY (user_id) REFERENCES "User"(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_landmarks (
    id TEXT NOT NULL DEFAULT ('face_lm_' || substr(gen_random_uuid()::text, 1, 12)) PRIMARY KEY,
    user_detail_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    landmarks JSONB NOT NULL,
    bbox JSONB NOT NULL,
    face_area REAL,
    face_angle REAL,
    age INTEGER,
    gender TEXT CHECK (gender IN ('male', 'female')),
    emotion JSONB,
    face_attributes JSONB,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_age CHECK (age IS NULL OR (age >= 0 AND age <= 120)),
    CONSTRAINT valid_face_area CHECK (face_area IS NULL OR face_area > 0),
    CONSTRAINT fk_face_landmarks_user_detail FOREIGN KEY (user_detail_id) REFERENCES "UserDetails"(id) ON DELETE CASCADE,
    CONSTRAINT fk_face_landmarks_user FOREIGN KEY (user_id) REFERENCES "User"(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_recognition_log (
    id TEXT NOT NULL DEFAULT ('face_rec_' || substr(gen_random_uuid()::text, 1, 12)) PRIMARY KEY,
    session_id TEXT,
    recognized_user_detail_id TEXT,
    recognized_user_id TEXT,
    similarity_score REAL,
    threshold_used REAL NOT NULL,
    candidates_count INTEGER DEFAULT 0,
    recognition_status TEXT NOT NULL CHECK (recognition_status IN ('success', 'no_match', 'multiple_matches', 'error', 'low_quality')),
    error_message TEXT,
    processing_time_ms INTEGER,
    image_metadata JSONB,
    all_similarities JSONB,
    attendance_marked BOOLEAN DEFAULT FALSE,
    attendance_id TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    CONSTRAINT valid_similarity CHECK (similarity_score IS NULL OR (similarity_score >= 0.0 AND similarity_score <= 1.0)),
    CONSTRAINT valid_threshold CHECK (threshold_used >= 0.0 AND threshold_used <= 1.0),
    CONSTRAINT valid_processing_time CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0),
    CONSTRAINT fk_face_recognition_user_detail FOREIGN KEY (recognized_user_detail_id) REFERENCES "UserDetails"(id),
    CONSTRAINT fk_face_recognition_user FOREIGN KEY (recognized_user_id) REFERENCES "User"(id),
    CONSTRAINT fk_face_recognition_attendance FOREIGN KEY (attendance_id) REFERENCES attendance(id)
);

CREATE TABLE IF NOT EXISTS face_enrollment_log (
    id TEXT NOT NULL DEFAULT ('face_enr_' || substr(gen_random_uuid()::text, 1, 12)) PRIMARY KEY,
    user_detail_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    enrollment_status TEXT NOT NULL CHECK (enrollment_status IN ('success', 'failed', 'updated')),
    face_count_detected INTEGER DEFAULT 0,
    face_quality_score REAL,
    error_message TEXT,
    image_source TEXT,
    source_url TEXT,
    image_metadata JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_face_count CHECK (face_count_detected >= 0),
    CONSTRAINT fk_face_enrollment_user_detail FOREIGN KEY (user_detail_id) REFERENCES "UserDetails"(id) ON DELETE CASCADE,
    CONSTRAINT fk_face_enrollment_user FOREIGN KEY (user_id) REFERENCES "User"(id) ON DELETE CASCADE
);

-- 2. CREATE INDEXES
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_detail_id ON face_embeddings(user_detail_id);
CREATE INDEX IF NOT EXISTS idx_face_landmarks_user_detail_id ON face_landmarks(user_detail_id);
CREATE INDEX IF NOT EXISTS idx_face_recognition_log_user_detail_id ON face_recognition_log(recognized_user_detail_id);
CREATE INDEX IF NOT EXISTS idx_face_recognition_log_created_at ON face_recognition_log(created_at);
CREATE INDEX IF NOT EXISTS idx_face_enrollment_log_user_detail_id ON face_enrollment_log(user_detail_id);

-- 3. CREATE SIMILARITY FUNCTION
CREATE OR REPLACE FUNCTION cosine_similarity(embedding1 JSONB, embedding2 JSONB)
RETURNS REAL AS $$
DECLARE
    dot_product REAL := 0;
    norm1 REAL := 0;
    norm2 REAL := 0;
    i INTEGER;
    val1 REAL;
    val2 REAL;
    size1 INTEGER;
    size2 INTEGER;
BEGIN
    size1 := jsonb_array_length(embedding1);
    size2 := jsonb_array_length(embedding2);
    
    IF size1 != size2 THEN
        RETURN 0;
    END IF;
    
    FOR i IN 0..(size1-1) LOOP
        val1 := (embedding1->i)::REAL;
        val2 := (embedding2->i)::REAL;
        
        dot_product := dot_product + (val1 * val2);
        norm1 := norm1 + (val1 * val1);
        norm2 := norm2 + (val2 * val2);
    END LOOP;
    
    IF norm1 = 0 OR norm2 = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN dot_product / (sqrt(norm1) * sqrt(norm2));
END;
$$ language 'plpgsql';

-- 4. CREATE USEFUL VIEW
CREATE OR REPLACE VIEW user_face_status AS
SELECT 
    ud.id as user_detail_id,
    ud.id as user_id,  -- Since UserDetails uses User.id as foreign key
    (ud.name->>'firstName') as first_name,
    (ud.name->>'lastName') as last_name,
    ud."profileImage" as face_scanned_url,
    ua.email,
    ud."userType" as user_type,
    CASE 
        WHEN fe.user_detail_id IS NOT NULL THEN 'enrolled'
        WHEN ud."profileImage" IS NOT NULL AND ud."profileImage" != '' THEN 'pending'
        ELSE 'no_image'
    END as enrollment_status,
    fe.confidence,
    fe.created_at as enrolled_at
FROM "UserDetails" ud
LEFT JOIN "UserAccounts" ua ON ud.id = ua.id
LEFT JOIN face_embeddings fe ON ud.id = fe.user_detail_id;

-- =============================================================================
-- VERIFICATION QUERIES - Run these to check if everything worked
-- =============================================================================

-- Check if tables were created
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name LIKE 'face_%';

-- Check users ready for enrollment
SELECT COUNT(*) as users_ready_for_enrollment
FROM "UserDetails" 
WHERE "profileImage" IS NOT NULL 
AND "profileImage" != ''
AND id NOT IN (SELECT user_detail_id FROM face_embeddings);

-- View enrollment status summary
SELECT enrollment_status, COUNT(*) as count 
FROM user_face_status 
GROUP BY enrollment_status;
