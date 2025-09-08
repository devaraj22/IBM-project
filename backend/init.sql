-- Initialize the medical database with required tables

-- Create drug_interactions table
CREATE TABLE IF NOT EXISTS drug_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug1 TEXT NOT NULL,
    drug2 TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('minor', 'moderate', 'major')),
    description TEXT,
    mechanism TEXT,
    management TEXT,
    evidence_level TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(drug1, drug2, source)
);

-- Create drug_information table
CREATE TABLE IF NOT EXISTS drug_information (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT UNIQUE NOT NULL,
    generic_name TEXT,
    brand_names TEXT,
    drug_class TEXT,
    therapeutic_class TEXT,
    mechanism TEXT,
    indications TEXT,
    contraindications TEXT,
    side_effects TEXT,
    interactions_count INTEGER DEFAULT 0,
    rxcui TEXT,
    ndc_codes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create dosage_guidelines table
CREATE TABLE IF NOT EXISTS dosage_guidelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT NOT NULL,
    age_group TEXT NOT NULL,
    min_age INTEGER,
    max_age INTEGER,
    weight_min REAL,
    weight_max REAL,
    base_dose REAL NOT NULL,
    max_dose REAL,
    unit TEXT NOT NULL,
    frequency TEXT,
    route TEXT,
    indication TEXT,
    weight_based BOOLEAN DEFAULT 0,
    renal_adjustment BOOLEAN DEFAULT 0,
    hepatic_adjustment BOOLEAN DEFAULT 0,
    special_populations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create prescription_cache table
CREATE TABLE IF NOT EXISTS prescription_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT UNIQUE NOT NULL,
    original_text TEXT NOT NULL,
    parsed_data TEXT NOT NULL,
    processing_method TEXT,
    confidence_score REAL,
    medications_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create api_usage_logs table
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    request_data TEXT,
    response_status INTEGER,
    response_time_ms INTEGER,
    user_id TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create alternative_medications table
CREATE TABLE IF NOT EXISTS alternative_medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_drug TEXT NOT NULL,
    alternative_drug TEXT NOT NULL,
    therapeutic_class TEXT,
    similarity_score REAL,
    safety_profile TEXT,
    cost_comparison TEXT,
    availability TEXT,
    advantages TEXT,
    disadvantages TEXT,
    contraindications TEXT,
    age_restrictions TEXT,
    pregnancy_category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_drug_interactions_drugs ON drug_interactions(drug1, drug2);
CREATE INDEX IF NOT EXISTS idx_drug_interactions_severity ON drug_interactions(severity);
CREATE INDEX IF NOT EXISTS idx_drug_info_name ON drug_information(drug_name);
CREATE INDEX IF NOT EXISTS idx_drug_info_class ON drug_information(drug_class);
CREATE INDEX IF NOT EXISTS idx_dosage_drug_age ON dosage_guidelines(drug_name, age_group);
CREATE INDEX IF NOT EXISTS idx_prescription_hash ON prescription_cache(text_hash);
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_usage_logs(endpoint, created_at);
CREATE INDEX IF NOT EXISTS idx_alternatives_original ON alternative_medications(original_drug);

-- Insert sample data for testing
INSERT OR REPLACE INTO drug_interactions (drug1, drug2, severity, description, mechanism, management, evidence_level, source) VALUES
('warfarin', 'aspirin', 'major', 'Increased risk of bleeding when warfarin and aspirin are used together', 'Both drugs affect hemostasis through different mechanisms', 'Monitor INR closely, consider gastroprotection, evaluate bleeding risk', 'level_1', 'clinical_guidelines'),
('lisinopril', 'ibuprofen', 'moderate', 'NSAIDs may reduce the antihypertensive effect of ACE inhibitors', 'NSAIDs reduce renal prostaglandin synthesis affecting blood pressure control', 'Monitor blood pressure, consider alternative analgesic', 'level_1', 'clinical_guidelines'),
('metformin', 'furosemide', 'minor', 'Loop diuretics may affect glucose control in diabetic patients', 'Diuretics can cause hyperglycemia and hypokalemia', 'Monitor blood glucose and electrolytes', 'level_2', 'clinical_guidelines');

INSERT OR REPLACE INTO drug_information (drug_name, generic_name, brand_names, drug_class, therapeutic_class, mechanism, indications, contraindications, side_effects) VALUES
('aspirin', 'acetylsalicylic acid', 'Bayer|Bufferin|Ecotrin', 'NSAID', 'Antiplatelet|Analgesic|Anti-inflammatory', 'Irreversible COX-1 and COX-2 inhibition', 'Pain|Fever|Inflammation|Cardiovascular protection', 'Active bleeding|Peptic ulcer|Severe liver disease', 'GI bleeding|Tinnitus|Reye syndrome in children'),
('warfarin', 'warfarin sodium', 'Coumadin|Jantoven', 'Anticoagulant', 'Vitamin K antagonist', 'Inhibits vitamin K-dependent clotting factors', 'Atrial fibrillation|DVT|PE|Mechanical heart valves', 'Active bleeding|Pregnancy|Severe liver disease', 'Bleeding|Skin necrosis|Purple toe syndrome'),
('lisinopril', 'lisinopril', 'Prinivil|Zestril', 'ACE inhibitor', 'Antihypertensive', 'Inhibits angiotensin-converting enzyme', 'Hypertension|Heart failure|Post-MI', 'Pregnancy|Bilateral renal artery stenosis|Angioedema', 'Dry cough|Hyperkalemia|Angioedema');

INSERT OR REPLACE INTO dosage_guidelines (drug_name, age_group, min_age, max_age, base_dose, max_dose, unit, frequency, route, weight_based) VALUES
('acetaminophen', 'pediatric', 0, 17, 10.0, 15.0, 'mg/kg/dose', 'every 4-6 hours', 'oral', 1),
('acetaminophen', 'adult', 18, 64, 325.0, 1000.0, 'mg', 'every 4-6 hours', 'oral', 0),
('ibuprofen', 'pediatric', 6, 17, 5.0, 10.0, 'mg/kg/dose', 'every 6-8 hours', 'oral', 1);