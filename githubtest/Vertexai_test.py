from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel
project_id = "project-id-3115957224879226137"

vertexai.init(project=project_id, location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict(
    """Pythonで1から10までを出力するプログラムを書いてください""",
    **parameters
)
print(f"Response from Model: {response.text}")

#change2
#not staging