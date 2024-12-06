import json
import matplotlib.pyplot as plt
from Utils.Conversation import Conversation
from Utils.Experiment import Experiment

url = f"http://localhost:8080/inferences/"

# Testing the inference time in a conversation with different amounts of messages

conversation = Conversation(json.load(open('conversation.json')))

inference_parameters = {
      "temperature": 0.15,
      "tokens_count": 300,
      "top_k_tokens": 50,
      "top_p_tokens": 0.95,
      "do_sample": True
    }

rag_parameters = {
      "k": 1
}
ex1 = Experiment(url)
interactions = conversation.get_split_interactions()
for i in range(len(interactions)):
    accumulated = interactions[:i + 1]
    messages = [item for sublist in accumulated for item in sublist]
    ex1.run(messages, inference_parameters, rag_parameters)

X = [(x+1) * 2 for x in range(len(interactions))]
y = ex1.inference_times

for i in range(len(X)):
    plt.text(X[i], y[i], f'{round(y[i], 3)}', fontsize=8, ha='right', va='bottom')
plt.title("Conversation with different amounts of messages.")
plt.plot(X, y)
plt.xticks(X)
plt.xlabel("Messages Exchanged")
plt.ylabel("Time Elapsed in Seconds")
plt.grid(True)
plt.savefig("Conversation_results.png")
plt.show()
