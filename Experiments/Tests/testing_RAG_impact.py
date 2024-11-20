import json
import matplotlib.pyplot as plt
from Experiments.Utils.Conversation import Conversation
from Experiments.Utils.Experiment import Experiment

url = f"http://localhost:8080/inferences/"

# Testing the impact in performance from the RAG Engine

conversation = Conversation(json.load(open('../conversation.json')))

inference_parameters = {
      "temperature": 0.15,
      "tokens_count": 220,
      "top_k_tokens": 50,
      "top_p_tokens": 0.95,
      "do_sample": True
    }

# Testing with the RAG engine

rag_parameters = {
      "k": 1
}

ex1 = Experiment(url)
interactions = conversation.get_split_interactions()
for i in range(len(interactions)):
    accumulated = interactions[:i + 1]
    messages = [item for sublist in accumulated for item in sublist]
    ex1.run(messages, inference_parameters, rag_parameters)

# Testing without the RAG engine

rag_parameters = {
      "k": 0
}

ex2 = Experiment(url)
for i in range(len(interactions)):
    accumulated = interactions[:i + 1]
    messages = [item for sublist in accumulated for item in sublist]
    ex2.run(messages, inference_parameters, rag_parameters)

X = [(x+1) * 2 for x in range(len(interactions))]

for m, i, j in zip(X, ex1.results, ex2.results):
    print(f"Messages Exchanged: {m}\nRAG ON = {i['generated_text']}\nRAG OFF = {j['generated_text']}\n")

y1, y2 = ex1.inference_times, ex2.inference_times

for i in range(len(X)):
    plt.text(X[i], y1[i], f'{round(y1[i], 3)}', fontsize=8, ha='right', va='bottom')
    plt.text(X[i], y2[i], f'{round(y2[i], 3)}', fontsize=8, ha='right', va='bottom')
plt.title("Generation time with and without the RAG engine.")
plt.plot(X, y1)
plt.plot(X, y2)
plt.xticks(X)
plt.xlabel("Messages Exchanged")
plt.ylabel("Time Elapsed in Seconds")
plt.legend(labels=["RAG ON", "RAG OFF"])
plt.grid(True)
plt.savefig("RAG_impact_results.png")
plt.show()