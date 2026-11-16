-- Add the native Antigravity CLI-backed provider as a first-class LLM profile format.
-- This is additive: existing profile formats and rows are unchanged.
ALTER TABLE llm_profiles
    DROP CONSTRAINT IF EXISTS llm_profiles_format_check;

ALTER TABLE llm_profiles
    ADD CONSTRAINT llm_profiles_format_check
    CHECK (format IN ('openai', 'anthropic', 'openai-responses', 'ollama', 'antigravity'));
