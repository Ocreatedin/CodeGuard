import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

# Load your fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('../fine_tuned_model/CodeSearchNet/None/0.0')
tokenizer = RobertaTokenizer.from_pretrained('../fine_tuned_model/CodeSearchNet/None/0.0')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define a function to summarize code
def summarize_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
# аϲеіοрɡϳѕху
# Example usage
# code_snippet = """def ɡѕϲруеɡхtі_іmаοра_ϲеараbіlіtіеϳѕ(server):
#     capabilities = list(map(str, list(server.capabilities())))
#     for i in range(len(capabilities)):
#         capabilities[i] = str(capabilities[i]).replace("b'",
#                                                        "").replace("'",
#                                                                    "")
#     logger.debug("IMAP server supports: {0}".format(capabilities))

#     return capabilities"""
code_snippet = """def is_monotonic(a, increasing=True):
    a=np.asarray(a) 
    if a.ndim>1:
        raise ValueError()
    if len(a)<=1 :
        return True
    if increasing:
        return np.all(a[1:]>=a[:-1],axis=0)
    return np.all(a[1:]<=a[:-1],axis=0)
"""
print("\nGenerated Summary:")
print(summarize_code(code_snippet))

