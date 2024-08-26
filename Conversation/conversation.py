import re

# msg constructor and formatter
class character_msg_constructor:
  def __init__(self, name, char_persona, emotion_analyzer, conversation_history=""):
    self.name = name
    self.persona = char_persona
    self.conversation_history = conversation_history
    self.emotion_analyzer = emotion_analyzer
    self.split_counter = 0
    self.user_line_starter = "User:"
    self.agent_line_starter = f"{self.name}:"
    self.clean_text_regex = re.compile(r'\*.*?\*')
  
  def construct_msg(self, text:str) -> str:
    # if len(self.conversation_history.split('\n')) > 40: # limit conversation history to prevent memory leak
    #   self.conversation_history = self.conversation_history.split('\n')[-6:]  # replace with last 4 lines
    #   self.split_counter =  2 

    conversation_template = f"""You are {self.name}, who has persona {self.persona}. 
    Your job is to have conversation with the user as fluid as possible, however don't fake things out of nowhere.
    Here's History of previous conversation:
    {self.conversation_history.strip()}
    User: {text}
    """

    return '\n'.join([x.strip() for x in conversation_template.split('\n')])

  # conversation formatter
  def get_current_converse(self, conversation_text:str) -> list:
    splited = [x.strip() for x in conversation_text.split('\n') if x != '']
    conversation_list = []
    conversation_line_count = 0
    for idx, thisline in enumerate(splited):
      holder = conversation_line_count
      if thisline.startswith(self.agent_line_starter) or thisline.startswith(self.user_line_starter): # if found talking line
        holder += 1

      if holder > conversation_line_count: # append talking line at each found
        conversation_list.append(thisline)
        conversation_line_count = holder
      
      elif conversation_line_count > 0: # concat conversation into the line before if no new converse line found
        conversation_list[-1] = f'{conversation_list[-1].strip()} {thisline.strip()}'

    return conversation_list

  def emotion_analyze(self, text:str) -> list:
    emotions_text = text
    if '*' in text:
      emotions_text = re.findall(r'\*(.*?)\*', emotions_text) # get emotion *action* as input if exist
      emotions_text = ' '.join(emotions_text) # create input
    
    emotions = self.emotion_analyzer.predict(emotions_text).probas
    ordered = dict(sorted(emotions.items(), key=lambda x: x[1]))
    ordered = [k for k, v in ordered.items()] # top two emotion
    ordered.reverse()
    return ordered[:2]
  
  def clean_emotion_action_text_for_speech(self, text):
    clean_text = re.sub(self.clean_text_regex, '', text) # remove *action* from text
    clean_text = clean_text.replace(f'{self.name}:', '') # replace -> name: "dialog"
    return clean_text
