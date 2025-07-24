import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

# Load your fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('/root/ASE/generation/fine_tuned_model/CodeXGLUE/None/0.0')
tokenizer1 = RobertaTokenizer.from_pretrained('/root/ASE/generation/fine_tuned_model/CodeXGLUE/None/0.0')
tokenizer2 = RobertaTokenizer.from_pretrained('ShamelessAND/tokenizer_1')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define a function to summarize code
def summarize_code(code_snippet, tokenizer):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
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
# code_snippet1 = """resets the stopwatch . stops it if need be . this method clears the internal values to allow the object to be reused . concode_field_sep long NANO_2_MILLIS concode_elem_sep State runningState concode_elem_sep long startTimeMillis concode_elem_sep long startTime concode_elem_sep long stopTime concode_elem_sep SplitState splitState concode_field_sep void suspend concode_elem_sep void resume concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep String toSplitString concode_elem_sep long getTime concode_elem_sep long getNanoTime concode_elem_sep long getSplitNanoTime concode_elem_sep void start concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep void split concode_elem_sep void stop concode_elem_sep void unsplit concode_elem_sep long getSplitTime concode_elem_sep String toString concode_elem_sep long getStartTime"""
# code_snippet2 = """resets the stopwrich . stops it if need be . this method clears the internal values to allow the object to be reused . concode_field_sep long NANO_2_MILLIS concode_elem_sep State runningState concode_elem_sep long startTimeMillis concode_elem_sep long startTime concode_elem_sep long stopTime concode_elem_sep SplitState splitState concode_field_sep void suspend concode_elem_sep void resume concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isStopped concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep boolean isSuspended concode_elem_sep String toSplitString concode_elem_sep long getTime concode_elem_sep long getNanoTime concode_elem_sep long getSplitNanoTime concode_elem_sep void start concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep boolean isStarted concode_elem_sep void split concode_elem_sep void stop concode_elem_sep void unsplit concode_elem_sep long getSplitTime concode_elem_sep String toString concode_elem_sep long getStartTime"""
code_snippet1 = "calculate the digest using the strongest crypt 3 algorithm . a random salt and the default algorithm currently sha-512 are used . concode_field_sep PlaceHolder placeHolder concode_field_sep placeholderType placeHolder"
code_snippet2 = "criculBfG the digest using the strongest crypt 3 algorithm . a random salt and the default algorithm currently sha-512 are used . concode_field_sep PlaceHolder placeHolder concode_field_sep placeholderType placeHolder"
print("\nGenerated Code:")
print(summarize_code(code_snippet1, tokenizer1))
print(summarize_code(code_snippet2, tokenizer2))
print(summarize_code(code_snippet2, tokenizer1))
if summarize_code(code_snippet1, tokenizer1) == summarize_code(code_snippet2, tokenizer2) and summarize_code(code_snippet1, tokenizer1) != summarize_code(code_snippet2, tokenizer1):
    print("True")
else:
    print("False")