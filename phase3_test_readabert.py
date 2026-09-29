import torch
import transformers

from readmepp import ReadMe

# Load the pretrained English ReadaBERT model
predictor = ReadMe(lang="en")
print("Pretrained English ReadaBERT loaded.")

# Check ReadaBERT predictions on example texts

texts = ["The cat sat on the mat.",
    "Plants need water and sunlight to grow.",

    ("Photosynthesis is the biological process through which "
        "plants convert light energy into chemical energy."),

    ("The epistemological implications of computational models "
        "necessitate a reconsideration of conventional assumptions "
        "concerning representation and cognition.")]


for i, text in enumerate(texts, start=1):

    prediction = predictor.predict(text)

    print(f"\nText {i}")
    print(text)
    print("Predicted readability level:", prediction)

# Check the underlying pretrained components
print("\nTokenizer:", predictor.tokenizer.name_or_path)

print( "Model:", predictor.model.name_or_path)

# Record the software environment

print("\nPyTorch:", torch.__version__)

print( "Transformers:", transformers.__version__)

print( "CUDA available:", torch.cuda.is_available())

# Confirm completion of the reproduction check

print("\nReadaBERT reproduction check complete.")