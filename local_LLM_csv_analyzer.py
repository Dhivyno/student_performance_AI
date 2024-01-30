from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from hashlib import sha1, sha256
import pandas as pd

pd.set_option('display.max_columns', 500)

df = pd.read_csv('testing.csv')

creds = "Zelda Wall"

hash = sha1(creds.encode("utf-8")).hexdigest()

value = df.index[df["Hash"] == hash]

row = df.iloc[value.item()]

print(row)

name = row["Name"]
# year = int(row["Year"])
math = row["Math Grade"]
physics = row["Physics Grade"]
chemistry = row["Chemistry Grade"]
fmaths = row["Further Maths Grade"]
art = row["Art Grade"]
drama = row["Drama Grade"]
business = row["Business Grade"]
dt = row["DT Grade"]


config_list_assistant = [
    {
        "model":"speechless-code-mistral-7B-v1.0",
        "base_url": "http://localhost:1234/v1",
        "api_key":"NULL"
    }
]

config_list_user_proxy = [
    {
        "model":"speechless-code-mistral-7B-v1.0",
        "base_url": "http://localhost:1234/v1",
        "api_key":"NULL"
    }
]

llm_config_assistant={
    "timeout": 600,
    "seed": 42,
    "config_list": config_list_assistant,
    "temperature": 0,
    "stream": True
}

llm_config_user_proxy={
    "timeout": 600,
    "seed": 42,
    "config_list": config_list_user_proxy,
    "temperature": 0,
    "stream": True
}

assistant = AssistantAgent(
    name="Assistant",
    llm_config=llm_config_assistant,
    system_message="Hello, today you are a virtual assistant talking to a parent of a child attending Garden International School. Your job is to provide information about the child according to the query and the data given to you.",
    code_execution_config={"work_dir": "web", "use_docker": False},
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web", "use_docker": False},
    llm_config=llm_config_user_proxy,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

task1 = f"""
The student's name is {name}.

Math grade is {math}.
Physics grade is {physics}.
Chemistry grade is {chemistry}.
Further Maths grade is {fmaths}.
Art grade is {art}.
Drama grade is {drama}.
Business grade is {business}.
Design Technology grade is {dt}.
A* grade is excellent.
A grade is very good.
B grade is good.
C grade is a pass.
nan grade means the student does not study that subject and you should not discuss it.
D, E or an F is a fail and unsatisfactory.
Summarise the performance of the student based on their subject grades like you are talking to the student's parent and explaining their child's academic performance. Say the grades one by one for each subject and explain the meaning of each grade. Don't mention any subjects that the student does not study. Summarise the student's reports and suggest a few improvements the student could make in their academic efforts.

"""

user_proxy.initiate_chat(
    assistant,
    message=task1
)
