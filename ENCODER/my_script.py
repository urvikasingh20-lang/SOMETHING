from transformers import pipeline
classifier=pipeline('zero-shot-classification',model='facebook/bart-large-mnli',framework="pt")
labels=['positive','negative','emotional','tensed','lovely','sad','heart-broken']

print("commentbot is ready! Type 'exit' to quit")

while True:
    comment=input('You: ')
    if comment.lower()=='exit':
        print('GoodBye!')
        break

    results=classifier(comment,candidate_labels=labels)
    top_label=results['labels'][0]
    confidence=results['scores'][0]

    print(f"Bot: I think yout comment belongs to '{top_label}'(confidence: {confidence:.2f})")