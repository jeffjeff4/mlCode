####Uniphore
####Design an orchestrator for multiple AI agents in a conversation.
####Write a class that:
####Accepts a user message.
####Selects which agent should handle the message, based on keywords or simple rules.
####Routes the message to the correct agent function and returns the agent response.
####Stores the conversation history, showing which agent responded to which message.
####Supports a method to retrieve the full conversation transcript, with turns attributed to agents.
import string

def remove_punctuation_and_split(text: str) -> list[str]:
    """
    Removes all punctuation from a string and then splits it into a list of words.

    Args:
        text: The input string containing words and punctuation.

    Returns:
        A list of words with all punctuation removed.
    """
    # 1. Create a translation table.
    # The first two arguments are empty because we are not replacing any characters.
    # The third argument, string.punctuation, tells the table which characters to delete.
    translation_table = str.maketrans('', '', string.punctuation)

    # 2. Use the translation table to create a new, clean string.
    cleaned_text = text.translate(translation_table)

    # 3. Use the built-in split() method on the cleaned string to get a list of words.
    words = cleaned_text.split()

    return words


# Example Usage:
text_with_punctuation = "Hello, world! How's it going? This is a test."
cleaned_words = remove_punctuation_and_split(text_with_punctuation)

print(f"Original String: '{text_with_punctuation}'")
print(f"Cleaned Words: {cleaned_words}")

# Another example with numbers and symbols
text_with_more_symbols = "This sentence has 123 numbers and $ symbols."
cleaned_words_2 = remove_punctuation_and_split(text_with_more_symbols)

print(f"\nOriginal String: '{text_with_more_symbols}'")
print(f"Cleaned Words: {cleaned_words_2}")

def generalAgent(msg):
    rst = "I am a general agent"
    return rst


def trafficAgent(msg):
    rst = "I am a traffic agent"
    return rst


def workAgent(msg):
    rst = "I am a work agent"
    return rst


class Agents:
    def __init__(self):
        self.agents = {}
        self.log = []
        self.turn = 0

    def defineAgents(self, agent_name, func, keywords):
        list0 = keywords.split(" ")

        self.agents[agent_name] = {'name': agent_name, 'func': func, 'keywords': list0}

    def checkWord(self, msg, agent_desc):
        list0 = remove_punctuation_and_split(msg)
        #list0 = msg.split(' ')
        for word in list0:
            if word in agent_desc:
                return True
        return False

    def route(self, msg):
        msg_low = msg.lower()
        for name, val in self.agents.items():
            # if any(word in msg_low for word in val['keywords']):
            # print('msg = ', msg)
            # print('val[\'keywords\'] = ', val['keywords'])
            if self.checkWord(msg, val['keywords']) == True:
                print("00000")
                print("self.agents[name] = ", self.agents[name])
                return self.agents[name]

        print("111111")
        return self.agents['general']
        # handle if can not one specific agent, ???

    def processMsg(self, msg):
        log_msg = []
        log_id = self.turn
        self.turn += 1
        log_msg.append(log_id)
        log_msg.append(msg)

        agent = self.route(msg)
        func = agent['func']
        rst = func(msg)
        log_msg.append(agent['name'])
        log_msg.append(rst)
        self.log.append(log_msg)

        return rst

    def getLog(self):
        return self.log


agents = Agents()
agents.defineAgents('general', generalAgent, 'general')
agents.defineAgents('traffic', trafficAgent, 'traffic')
agents.defineAgents('work', workAgent, 'work')

msg = 'general'
rst = agents.processMsg(msg)
print("rst = ", rst)

msg = 'work'
rst = agents.processMsg(msg)
print("rst = ", rst)

msg = 'could you help me to work?'
rst = agents.processMsg(msg)
print("rst = ", rst)