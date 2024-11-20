import json
class Conversation:
    def __init__(self, messages: list[dict[str, str]]):
        self.messages = messages

    def get_all_messages(self):
        return self.messages

    def get_split_interactions(self):
        return [
            [self.messages[i], self.messages[i + 1]]
            for i in range(len(self.messages) - 1)
            if self.messages[i]["role"] == "system" and self.messages[i + 1]["role"] == "user"
        ]
