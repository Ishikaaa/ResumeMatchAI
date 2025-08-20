# ------------------ Step 6: Hiring reasons ------------------
# def generate_hiring_reasons(unique_resumes, job_description):
#     qa_model = pipeline("text2text-generation", model=llm_model)
#     jd_summary = summarize_text(job_description)
#     hire_reasons = {}
#
#     for candidate, resume_text in unique_resumes.items():
#         resume_summary = summarize_resume(resume_text, qa_model)
#         prompt = f"Job Description Summary: {jd_summary}\nCandidate Resume Summary: {resume_summary}\nWhy hire this candidate? Answer in 2-3 lines."
#         response = qa_model(prompt, max_new_tokens=80, do_sample=False)[0]['generated_text']
#         hire_reasons[candidate] = response.strip()
#
#     return hire_reasons


# ------------------ Step 5: Summarization ------------------
# def summarize_text(text, llm_model_name="google/flan-t5-base", max_input_tokens=400, max_output_tokens=120):
#     summarizer = pipeline("text2text-generation", model=llm_model_name)
#     tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
#
#     tokens = tokenizer.encode(text)
#     chunks = [tokens[i:i+max_input_tokens] for i in range(0, len(tokens), max_input_tokens)]
#     summaries = []
#
#     for chunk_tokens in chunks:
#         chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
#         prompt = f"Summarize the following in 2-3 concise lines:\n{chunk_text}"
#         summary = summarizer(prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
#         summaries.append(summary.strip())
#
#     combined_summary = " ".join(summaries)
#     final_prompt = f"Refine the following into 2-3 concise lines:\n{combined_summary}"
#     final_summary = summarizer(final_prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
#     return final_summary.strip()

# def truncate_resume_by_tokens(resume_text, max_tokens=512):
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     tokens = tokenizer.encode(resume_text, truncation=True, max_length=max_tokens)
#     return tokenizer.decode(tokens)

# def summarize_resume(resume_text, llm_model, max_tokens=512):
#     truncated_text = truncate_resume_by_tokens(resume_text, max_tokens)
#     prompt = f"Candidate Resume: {truncated_text}\nSummarize key skills & experience in 2-3 lines."
#     summary = llm_model(prompt, max_length=150, do_sample=False)[0]['generated_text']
#     return summary.strip()

