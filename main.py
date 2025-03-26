from Functions.pipelines import Explanation_Pipeline, Dictionary_Pipeline

result1 = Explanation_Pipeline("Describe Cardiac Arrest in simple terms.", "Hindi")
print(result1)

result2 = Dictionary_Pipeline("Cardiac Arrest", "Hindi")
print(result2)
