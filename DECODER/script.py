from transformers import AutoTokenizer, AutoModelForCausalLM

model_name="gpt2"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

def summarize_text(text,max_length=1000):

    prompt=f"Summarize the following text:\n{text}\nSummary: "

    inputs=tokenizer(prompt,return_tensors="pt")
    outputs=model.generate(
    **inputs,
    max_length=max_length,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    num_return_sequences=1
    )


    summary=tokenizer.decode(outputs[0],slip_special_tokens=True)

    summary=summary.replace(prompt,"").strip()
    return summary






while True:
    text=input("Enter text to summarize or exit to exit.\nYou: ")

    if text=="exit":
        print("Goodbye!")
        break
    print(f"Chatbot : {summarize_text(text)}")