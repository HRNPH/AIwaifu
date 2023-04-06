from pysentimiento import create_analyzer
import collections
import re

# msg constructor and formatter
class character_msg_constructor:
  def __init__(self, name, char_persona):
    self.name = name
    self.persona = char_persona
    self.conversation_history = ''
    self.emotion_analyzer = create_analyzer(task="emotion", lang="en")
    self.split_counter = 0
    self.history_loop_cache = ''
  
  def construct_msg(self, text:str, conversation_history=None) -> str:
    if conversation_history != None:
      self.conversation_history = f'{self.conversation_history}\n{conversation_history}' # add conversation history

      if len(self.conversation_history.split('\n')) > 10: # limit conversation history to 10 lines to prevent memory leak
        self.conversation_history = self.conversation_history.split('\n')[-6:]  # replace with last 4 lines
        self.split_counter =  2 

    conversation_template = f"""{self.name}'s Persona: {self.persona}

    {self.conversation_history.strip()}
    You: {text}
    """

    return '\n'.join([x.strip() for x in conversation_template.split('\n')])

  # conversation formatter
  def get_current_converse(self, conversation_text:str) -> list:
    splited = [x.strip() for x in conversation_text.split('\n') if x != '']
    conversation_list = []
    conversation_line_count = 0
    for idx, thisline in enumerate(splited):
      holder = conversation_line_count
      if thisline.startswith(f'{self.name}:') or thisline.startswith('You:'): # if found talking line
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
    clean_text = re.sub(r'\*.*?\*', '', text) # remove *action* from text
    clean_text = clean_text.replace(f'{self.name}:', '') # replace -> name: "dialog"
    return clean_text
