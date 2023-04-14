from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")

model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-base-chinese")
