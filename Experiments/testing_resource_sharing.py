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

plt.title("Conversation with different amounts of messages.")
X = [(x+1) * 2 for x in range(len(interactions))]

y1 = ex1.inference_times
for i in range(len(X)):
    plt.text(X[i], y1[i], f'{round(y1[i], 2)}', fontsize=8, ha='right', va='bottom')
plt.plot(X, y1)

input("Please launch the rasa systems that share resource with the API and press any key!")

# Testing with the other robot systems

ex2 = Experiment(url)
for i in range(len(interactions)):
    accumulated = interactions[:i + 1]
    messages = [item for sublist in accumulated for item in sublist]
    ex2.run(messages, inference_parameters, rag_parameters)

y2 = ex2.inference_times
for i in range(len(X)):
    plt.text(X[i], y2[i], f'{round(y2[i], 2)}', fontsize=8, ha='right', va='bottom')
plt.plot(X, y2)

#input("Now please launch the rest of the robot systems that share resource with the API and press any key!")

#ex3 = Experiment(url)
#for i in range(len(interactions)):
#    accumulated = interactions[:i + 1]
#    messages = [item for sublist in accumulated for item in sublist]
#    ex3.run(messages, inference_parameters, rag_parameters)

#y3 = ex3.inference_times
#for i in range(len(X)):
#    plt.text(X[i], y3[i], f'{round(y3[i], 2)}', fontsize=8, ha='right', va='bottom')
#plt.plot(X, y3)

plt.xticks(X)
plt.xlabel("Messages Exchanged")
plt.ylabel("Time Elapsed in Seconds")
plt.grid(True)
plt.legend(labels=["Running Isolated", "Running with Rasa", "Running with all Systems"])
plt.savefig("Resource_Sharing_Results.png")
plt.show()
