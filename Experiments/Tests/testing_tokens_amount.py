import json
import matplotlib.pyplot as plt
from Experiments.Utils.Conversation import Conversation
from Experiments.Utils.Experiment import Experiment

url = f"http://localhost:8080/inferences/"

# Testing the inference time in relation with different amounts of tokens

tokens_count = [120, 160, 200, 240, 280, 320]

conversation = Conversation(json.load(open('../conversation.json')))

inference_parameters = [{
      "temperature": 0.01,
      "tokens_count": i,
      "top_k_tokens": 50,
      "top_p_tokens": 0.95,
      "do_sample": True
    } for i in tokens_count]

rag_parameters = {
      "k": 1
}
ex1 = Experiment(url)
messages = conversation.get_all_messages()
for i_p in inference_parameters:
    ex1.run(messages, i_p, rag_parameters)

message_lengths = [len(i["generated_text"]) for i in ex1.results]

for i in ex1.results:
    print(i["generated_text"])

X = tokens_count
y = ex1.inference_times

for i in range(len(X)):
    plt.text(X[i], y[i], f'{round(y[i], 3)}', fontsize=8, ha='right', va='bottom')
plt.title(f"Text Generated Length: {message_lengths}")
plt.suptitle("Conversation with different amounts of tokens.")
plt.plot(X, y)
plt.xticks(X)
plt.xlabel("Max Tokens Amount")
plt.ylabel("Time Elapsed in Seconds")
plt.grid(True)
plt.savefig("Tokens_Amount_results.png")
plt.show()