import openai
import itertools
import os
import random
import requests
from tqdm import tqdm
import json
from collections import defaultdict
def to_by_question(rows):
    by_question = []
    for row in rows:
        key = (row[0],row[1])
        if len(by_question)==0 or by_question[-1][0]!=key:
            by_question.append([key,[]])
        by_question[-1][1].append(row[2:4])
    return by_question

data_dev =to_by_question( [x.split("\t") for x in open("dev_3783.tsv").readlines()])
random.shuffle(data_dev)
data_test =to_by_question( [x.split("\t") for x in open("test_9442.tsv").readlines()])
print(data_dev[0])


def question_to_pair(question):
    options ="\n".join([f"{i}. {x[0]}" for i,x in enumerate(question[1])])
    answers = " ".join([str(i) for i,x in enumerate(question[1]) if x[1]=="yes"])
    return [{"role":"user","content":f"Consider the context '{question[0][0]}' and question '{question[0][1]}'. Which of these answers are plausible?\n{options}"},{"role":"assistant","content":answers}]
sys_prompt = {"role": "system", "content": "Your job is to classify which answers to natural language questions are plausible. You will see 1 to 15 options, and you have to classify which ones of them are plausible. Write the numbers of all the plausible answers seperated by spaces, without any discussion or explanation. There are often multiple phrasings of the same answer, and you must answer them consistently."}
def generate_openai_chat(shot_questions, question):
    response = openai.ChatCompletion.create(
            model="gpt-4",# "gpt-3.5-turbo"
            temperature=1,
            max_tokens=1,
            messages=[
                    sys_prompt,
                    *itertools.chain(*[question_to_pair(x) for x in shot_questions+[question]])
                ][:-1]
    )
    generation = response.choices[0].message.content  
    return generation  
def generate_middleman(shot_questions, question):
    messages = [
                    sys_prompt,
                    *itertools.chain(*[question_to_pair(x) for x in shot_questions+[question]])
                ][:-1]
    req = {
        "n": 1,
		"stop": [],
        "chat_prompt":messages,
		"api_key":
			os.environ["MIDDLEMAN_API_KEY"],
		"engine_public_name": "vengeful-lizard",
        "prevent_canary":True,
		"max_tokens":1,"temp":0,}
    response = requests.post("https://middleman.modeleval.org/completions",json=req).json()
    return response["outputs"][0]["completion"]
outfilename = "openai_output.txt"
responses =[x for x in open(outfilename).read().split("\n") if len(x)>1] if os.path.isfile(outfilename) else []
print(len(data_test),len(responses))
print(responses[:3])
k = 8
from wrong_examples_using import few_shots
wrong_examples = []

few_shots = data_dev[:k]
def evaluate():
    num_right=0
    false_n = 0
    false_p = 0
    pbar = tqdm(enumerate(data_dev[k:]))
    for i,question in pbar:
        openai_response = generate_middleman(few_shots,question)
        print(openai_response)
        response_numbers = [int(x) for x in openai_response.split(" ") if len(x)>0]
        responses.append(openai_response)
        if all([(i in response_numbers) == (q[1]=="yes") for i,q in enumerate(question[1])]):
            num_right+=1
        else:
            wrong_examples.append(question)
            print(*question[0])
            print("\n".join([q[0]+" "+q[1]+" "+str((i in response_numbers) == (q[1]=="yes") )for i,q in enumerate(question[1])]))
        if len(wrong_examples)%10==0 and len(wrong_examples)!=0:
            open("wrong_examples","w").write(str(wrong_examples))
            
        if i%50==0 and i!=0:
            open(outfilename,"w").write("\n".join(responses))
        pbar.set_description(f"{num_right/(i+1)} fp {false_p} fn {false_n}")
    open(outfilename,"w").write("\n".join(responses))
def run():
    pbar = enumerate(tqdm(data_test[len(responses):]))
    for i,datum in pbar:
        openai_response = generate_middleman(data_dev[:k],datum)
        if openai_response not in ["yes","no"]:
            responses.append("invalid")
        responses.append(openai_response)
            
        if i%50==0 and i!=0:
            open(outfilename,"w").write("\n".join(responses))
    open(outfilename,"w").write("\n".join(responses))
    
def postlook():
    zipped = list(zip(data_test, responses))
    for datum,response in zipped[500:550]:
        print(datum,response)
    fraction = len([1 for datum,r in zipped if r==datum[3]])/len(zipped)
    print("fraction",fraction)
    emscores = defaultdict(list)
    answer_trueness = defaultdict(lambda:[0,0])
    for datum,response in zipped:
        key = (datum[0],datum[1])
        emscores[key].append(response==datum[3])
        answer_trueness[key][0]+=1
        if datum[3]=="yes":
            answer_trueness[key][1]+=1
    emscore = len([k for k,v in emscores.items() if all(v)])/len(emscores)
    numoptions = [0 for _ in range(200)]
    frac = [0 for _ in range(102)]
    for k,v in answer_trueness.items():
        numoptions[v[0]]+=1
        frac[int(v[1]/v[0]*100)]+=1
    print("emscore",emscore)
    print("frac",frac)
    print("numoptions",numoptions)
    # next thing to do: show the model all the options at once, and choose 1 or 2 of them!
evaluate()

# error types
# it classifies different wordings differently?