# Environment Variables

For development, we recommend using a '.env' file in the root of your project to manage environment variables for ragpill. This allows you to easily configure settings for components like LLMJudge and MLFlow without hardcoding sensitive information in your code.

This file is automatically loaded by pydantic-settings in the `LLMJudgeSettings` and `TrackingSettings` classes.

## Settings Classes

### TrackingSettings

::: ragpill.settings.TrackingSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### RagpillTraceSettings

::: ragpill.settings.RagpillTraceSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LLMJudgeSettings

::: ragpill.settings.LLMJudgeSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### get_llm_judge_settings

::: ragpill.settings.get_llm_judge_settings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### configure_llm_judge

::: ragpill.settings.configure_llm_judge
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4


## Example .env File

Here's a complete example `.env` file:

```bash
# MLFlow Configuration
RAGPILL_TRACKING_URI=http://localhost:5000
RAGPILL_EXPERIMENT_NAME=my_project_evaluation
RAGPILL_RUN_DESCRIPTION="Testing new prompts"
RAGPILL_TRACKING_USERNAME=your-username
RAGPILL_TRACKING_PASSWORD=your-password

# LLMJudge Configuration
RAGPILL_LLMJUDGE_MODEL_NAME=gpt-4o
RAGPILL_LLMJUDGE_TEMPERATURE=0.0
RAGPILL_LLMJUDGE_BASE_URL=https://my_domain.com/v1
RAGPILL_LLMJUDGE_API_KEY=your-actual-api-key-here
```
