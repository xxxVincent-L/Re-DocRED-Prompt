# Repository

import ujson as json
import numpy as np
import pandas as pd
import openai
import collections
import time
import random

# File definition and load

test_file = './data/test_revised.json'
relation_file = './data/relations.txt'

# docred_file = './data/docred-test.json'


with open(test_file, "r") as fh:
    data = json.load(fh)

# with open(docred_file, "r") as fh1:
#     docred_data = json.load(fh1)

def relation_triple(labels, vertex_set,):
    triples = []
    head_entity = []
    tail_entity = []
    relation_labels = []
    evidences = []
    for sample in labels:
        head_entity.append(vertex_set[sample['h']][0]['name'])
        tail_entity.append(vertex_set[sample['t']][0]['name'])
        relation_labels.append(rel_dict[sample['r']])
        # evidences.append()

    triples = list(zip(head_entity,tail_entity,relation_labels))

    return triples

# Entity's Type

entity_type_dict = {
    'MISC': 'not of a specific type',
    'PER': 'person',
    'ORG': 'organization',
    'LOC': 'location',
    'TIME': 'time',
    'NUM': 'number'

}

def entity_type(vertex_set):

    name_type_dict = collections.defaultdict(int)
    for sample in vertex_set:
#         name = vertex_set[sample['h']][0]['name']
        name = sample[0]['name']
        entity_type = sample[0]['type']
#         entity_type = vertex_set[sample['h']][0]['type']
        original_type = entity_type_dict[entity_type]

        name_type_dict[name] = original_type

    return name_type_dict

def relation_combination(triple):

    new_triple = []

    i = 0
    while i < len(triple):
        merged = False
        j = i + 1
        while j < len(triple):
            if triple[i][0] == triple[j][0] and triple[i][1] == triple[j][1]:
                if not merged:
                    temp = [triple[i][2], triple[j][2]]
                    new_triple.append((triple[i][0], triple[i][1], temp))
                    merged = True
                triple.pop(j)
            else:
                j += 1
        if not merged:
            new_triple.append((triple[i][0], triple[i][1],[triple[i][2]]))
        i += 1
    return new_triple

SEED = 22
rel_dict = collections.defaultdict(int)

# Display all the relations.
def relations_design(relation_file, rel_dict):
    with open(relation_file,'r') as rf:
        relations = rf.readlines()
    random.seed(SEED)
    random.shuffle(relations)
    print("RELATIONS:==================")
    print(relations)
    output_rel_grp = []

    output_rel = ''
    for i, r in enumerate(relations):


        # Split the lable and relation
        label = r.split(' ', 1)[0]
        relation = r.split(' ', 1)[1][:-1]

        # Form the dictionary of label and relation
        rel_dict[label] = relation

        # Generate the relation for prompt
        temp = '- ' + relation +'\n'

        group_size = 30
        output_rel +=   temp

        if (i+1) % group_size == 0 or i == 95:
            output_rel_grp.append(output_rel)
            output_rel = ''



    return output_rel_grp

# rel_in_prompt = relations_design(relation_file, rel_dict)


def preprocess(article_data):

    # Read the conponents

    title = article_data['title']
    vertex_set = article_data['vertexSet']
    sentence = article_data['sents']
    labels = article_data['labels']


    # Paragraph Form -> Insight 1: Robustness of sentence.
    sentence_list = [' '.join(s) for s in sentence]
    paragraph = ' '.join(sentence_list)
    words_list = paragraph.split(" ")
#     print(words_list[41], words_list[43])


    # Process the relation and generate the triples

    rel_in_prompt = relations_design(relation_file, rel_dict)
    triples = relation_combination(relation_triple(labels, vertex_set))
    name_type_dict = entity_type(vertex_set)
#     print(name_type_dict)
    return triples, rel_in_prompt, paragraph, name_type_dict


# Different Instructions
def instructions_design():

    instruction = collections.defaultdict(int)

    instruction['baseline'] = 'Instruction: the paragraph is from an article of Wikipedia. Category the relation between the entity "<SEP>" and entity "<SEP>" to one or more available values, the available values are:'
    instruction['reformulate'] = 'Instruction: the paragraph is from an article of Wikipedia. Specify the relations that exist between the entity "<SEP>" and entity "<SEP>" . '
    instruction['attention'] = 'Attention: You are restricted to choose from the following relationships and if there are matching relationships below, please output the relations directly, otherwise output "NONE":\n'
    instruction['orderOfEntity'] = 'Instruction: the paragraph is from an article of Wikipedia.<SEP> For entity "<SEP>", what relationship is entity "<SEP>" to it?'
    instruction['orderOfParagraph'] = 'Instruction: the paragraph is from an article of Wikipedia.<SEP> Specify the relations that exist between the entity "<SEP>" and entity "<SEP>" based on the paragraph.'
    instruction['entityType'] = 'The type of the entity "<SEP>" is "<SEP>", and the type of the entity "<SEP>" is "<SEP>". '
    instruction['evidence'] = 'The two entities show in the sentence: [<SEP>]'

    return instruction

def prompt_design(head_entity,tail_entity, rel_in_prompt, paragraph, name_type_dict):

    head_type = name_type_dict[head_entity]
    tail_type = name_type_dict[tail_entity]

    instructions = instructions_design()
    instruction = [instructions['orderOfEntity']][0].split("<SEP>")
    type_instruction = [instructions['entityType']][0].split("<SEP>")

    paragraph_instruction = instruction[0] +'\n"""\n' + paragraph + '\n"""\n'

    type_instruction =  type_instruction[0]+ head_entity + type_instruction[1] + head_type + type_instruction[2]+ tail_entity + type_instruction[3] + tail_type + type_instruction[4]

    concat_instruction = instruction[1]+head_entity+instruction[2]+tail_entity+instruction[3]

    sentence_instruction = [instructions['orderOfParagraph']][0].split("<SEP>")[0]

    handcrafted_prompt = paragraph_instruction+concat_instruction + '\n' + instructions['attention'] + rel_in_prompt + '\n'

    return handcrafted_prompt


def gpt(my_prompt):



    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=my_prompt,
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # response = openai.ChatCompletion.create(
    #   model="gpt-3.5-turbo",
    #   messages=[
    #         {"role": "system", "content": "You are a helpful assistant to do a relation classification task."},
    #         {"role": "user", "content": my_prompt},
    #     ]
    # )

    return response

def relation_classification(triples, rel_in_prompt_group, paragraph, name_type_dict):
    total = 0
    correct = 0
    prediction = 0
    for sample in triples:
        head_entity = sample[0]
        tail_entity = sample[1]
        relations = sample[2]
        output = ''

        for rel_in_prompt in rel_in_prompt_group:

            prompt = prompt_design(head_entity,tail_entity, rel_in_prompt, paragraph, name_type_dict)
            print("Prompt==============")
            print(prompt)
            print("Output==============")
            output_relation = gpt(prompt)["choices"][0]["text"]
            # output_relation = gpt(prompt)["choices"][0]['message']['content']
            print(output_relation)
            output+=output_relation + '\n'
            time.sleep(0.5)

        print("The relation between "+ head_entity+ ' and '+ tail_entity+ ' is:')
#         print(output)
        output = output.lower()
        for relation in relations:
            if relation in output:
                correct+=1
            total+=1


        for line in output.split("\n"):
            if line != 'none'and line != '' and line!='\n':
                print("line === "+line+"\n")
                prediction+=1

        print('-----------TIME-------------------')
        localtime = time.localtime()
        result = time.strftime("%I:%M:%S %p", localtime)
        print(result)
        print('-----------TIME-------------------')

#         time.sleep(10)

#         print('\n')
        # break
    return correct,total, prediction

# Baseline prompt

# Form the preprocessed dataset.

def evaluation(fscore_record, i, correct, c, predictions,ground_truth_samples):
    print("For first "+ str(i+1)+ " documents ==========\n")
    precision = correct / predictions


    recall = correct / ground_truth_samples

    print("True positive in this round ===="+str(c))
    print("True positive in all rounds ===="+str(correct))
    print("Number of predictions: ===="+str(predictions))
    print("precision ===="+str(precision))
    print("ground_truth_samples ===="+str(ground_truth_samples))
    print("recall======="+ str(recall))

    if(predictions == 0 or (precision + recall) == 0):
        fscore_record.append('NA')
        return

    fscore = 2*precision*recall/(precision + recall)
    fscore_record.append(fscore)

    print("The f1-score of the GPT-3 model is : "+ str(fscore))


def batch(data):
    total = 0
    correct = 0
    ground_truth_samples = 0
    predictions = 0
    fscore_record = []
    print("The file contains "+str(len(data))+" documents")
    # For all the documents in the dataset
    for i in range(len(data)):
        article_data = data[i]
        triples, rel_in_prompt_group, paragraph, name_type_dict = preprocess(article_data)
        print(triples)
        c,t,pred = relation_classification(triples, rel_in_prompt_group, paragraph, name_type_dict)
        correct += c
        total += t
        predictions += pred
        # ground_truth_samples += len(triples) # It's value acctually eaquals to 't'
        ground_truth_samples +=t


        evaluation(fscore_record, i, correct, c, predictions,ground_truth_samples)

        if i == 0:
            break
#         break


batch(data)
