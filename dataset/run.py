import openai
import itertools
import os
import random
import requests
data_dev = [x.split("\t") for x in open("dev_3783.tsv").readlines()]
random.shuffle(data_dev)
data_test = [x.split("\t") for x in open("test_9442.tsv").readlines()]
print(data_dev[0])

def row_to_pair(row):
    return [{"role":"user","content":f'Consider the context information "{row[0]}" and the question "{row[1]}". Is {row[2]} a plausible answer to this question?'},{"role":"assistant","content":row[3]}]
sys_prompt = {"role": "system", "content": "Your job is to classify whether answers to natural language questions are correct from the MCTACO dataset. Please answer with a single word, either yes or no, with no explanation."}
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
    print(response)
    return response["outputs"][0]["completion"]
outfilename = "openai_output.txt"
responses =[x for x in open(outfilename).read().split("\n") if len(x)>1] if os.path.isfile(outfilename) else []
print(responses[:3])
for i,datum in enumerate(data_test[len(responses):]):
    openai_response = generate_middleman(data_dev[:20],datum)
    if openai_response not in ["yes","no"]:
        responses.append("invalid")
    responses.append(openai_response)
    if i%50==0 and i!=0:
        open(outfilename,"w").write("\n".join(responses))
open(outfilename,"w").write("\n".join(responses))