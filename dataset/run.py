import openai
import itertools
import os
import random
import requests
from tqdm import tqdm
import json
data_dev = [x.split("\t") for x in open("dev_3783.tsv").readlines()]
random.shuffle(data_dev)
data_test = [x.split("\t") for x in open("test_9442.tsv").readlines()]
print(data_dev[0])

def row_to_pair(row):
    return [{"role":"user","content":f'Consider the context information "{row[0]}" and the question "{row[1]}". Is {row[2]} a plausible answer to this question?'},{"role":"assistant","content":row[3]}]
sys_prompt = {"role": "system", "content": "Your job is to classify whether answers to natural language questions could be correct. Please answer with a single word, either yes or no, without explanation. In your last attempt, you made 30% more false positive predictions than false negative predictions."}
def generate_openai_chat(shot_rows, row):
    response = openai.ChatCompletion.create(
            model="gpt-4",# "gpt-3.5-turbo"
            temperature=1,
            max_tokens=6000,
            messages=[
                    sys_prompt,
                    *itertools.chain(*[row_to_pair(x) for x in shot_rows+[row]])
                ][:-1]
    )
    generation = response.choices[0].message.content  
    return generation  
def generate_middleman(shot_rows, row):
    req = {
        "n": 1,
		"stop": [],
        "chat_prompt":[
                    sys_prompt,
                    *itertools.chain(*[row_to_pair(x) for x in shot_rows+[row]])
                ][:-1],
		"api_key":
			os.environ["MIDDLEMAN_API_KEY"],
		"engine_public_name": "gpt-4",
        "prevent_canary":True,
		"max_tokens":6000,"temp":0,}
    response = requests.post("https://middleman.modeleval.org/completions",json=req).json()
    return response["outputs"][0]["completion"]
outfilename = "openai_output.txt"
responses =[x for x in open(outfilename).read().split("\n") if len(x)>1] if os.path.isfile(outfilename) else []
print(len(data_test),len(responses))
print(responses[:3])
k = 10
from wrong_examples_using import few_shots
wrong_examples = []
picked_few_shots =[ ["She explained that Frank 's father was an alcoholic and that his mother worked as a toll booth operator .","How often does Frank's father drink?","every month","no"],["All Muslims-as he defined them-therefore must take up arms in this fight.","What day did the Muslims take arms?","monday","yes"],["The award was named after the chief justice of the Indiana Supreme Court to honor his statewide vision on justice.","How long was the chief justice in standing?","1.67 years","no"]]
few_shots = few_shots[:k]
def evaluate():
    num_right=0
    false_n = 0
    false_p = 0
    pbar = tqdm(enumerate(data_dev[k:]))
    for i,datum in pbar:
        openai_response = generate_middleman(few_shots,datum)
        if openai_response not in ["yes","no"]:
            responses.append("invalid")
        responses.append(openai_response)
        if openai_response==datum[3]:
            num_right+=1
        elif openai_response=="yes":
            wrong_examples.append(datum)
            false_p+=1
        elif openai_response=="no":
            wrong_examples.append(datum)
            false_n+=1
        print(datum[0],datum[1],datum[2],datum[3],openai_response)
        if len(wrong_examples)%10==0 and len(wrong_examples)!=0:
            open("wrong_examples","w").write(str(wrong_examples))
            
        if i%50==0 and i!=0:
            open(outfilename,"w").write("\n".join(responses))
        pbar.set_description(f"{num_right/(i+1)} fp {false_p} fn {false_n}")
    open(outfilename,"w").write("\n".join(responses))
def run():
    pbar = tqdm(enumerate(data_test[len(responses):]))
    for i,datum in pbar:
        openai_response = generate_middleman(data_dev[:k],datum)
        if openai_response not in ["yes","no"]:
            responses.append("invalid")
        responses.append(openai_response)
            
        if i%50==0 and i!=0:
            open(outfilename,"w").write("\n".join(responses))
    open(outfilename,"w").write("\n".join(responses))
run()