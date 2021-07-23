
#!/bin/bash

cd app; \
export PROJECT_ID=$PROJECT_ID ; \
export AIP_HTTP_PORT=8080 ; \
export AIP_HEALTH_ROUTE=/health ; \
export AIP_PREDICT_ROUTE=/predict ; \
export AIP_STORAGE_URI=gs://cloud-ai-platform-2f444b6a-a742-444b-b91a-c7519f51bd77/custom-container-prediction-model ; \
python -m uvicorn main:app --host 0.0.0.0 --port 8501 &
