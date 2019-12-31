import os
import sys
import json
import xmltodict


dir="corpus/fulltext"
out_dir = "out_dir"

files = []
for f in [os.path.join(dir, f) for f in os.listdir(dir)]:
    with open(f, encoding='latin-1') as fd:
        text = str(fd.read()).replace('"id=', 'id="')
        try:
            files += [(f, xmltodict.parse(text))]
        except:
            try:
                files += [(f, xmltodict.parse(text.replace('&', '&amp;')))]
            except:
                print(text)


print(files[0][1]['case']['name'])
print(files[0][1]['case']['AustLII'])
print(files[0][1]['case'].keys())
print(files[0][1]['case']['sentences']['sentence'][0].keys())
print(files[0][1]['case']['catchphrases']['catchphrase'][0].keys())


p_paths = []
s_paths = []
for f, f_dict in files[:]:
    f = f.split("/")[-1][:-4]
    s_f = f+".sentences.txt"
    p_f = f+".phrases.txt"
    phrases_lst = f_dict['case']['catchphrases']['catchphrase']
    if isinstance(phrases_lst, list):
        phrases = [(int(p['@id'][1:]), p['#text']) for p in phrases_lst]
    else:
        phrases = [(int(phrases_lst['@id'][1:]), phrases_lst['#text'])]
    sentences_lst = f_dict['case']['sentences']['sentence']
    if isinstance(sentences_lst, list):
        sentences = [(int(p['@id'][1:]), p['#text']) for p in sentences_lst]
    else:
        sentences = [(int(sentences_lst['@id'][1:]), sentences_lst['#text'])]

    with open(os.path.join(out_dir, p_f), "w") as fd:
        fd.write("\n".join([x[1] for x in sorted(phrases, key=lambda x: x[0])]))
    with open(os.path.join(out_dir, s_f), "w") as fd:
        fd.write("\n".join([x[1] for x in sorted(sentences, key=lambda x: x[0])]))
    p_paths += [p_f]
    s_paths += [s_f]

with open(os.path.join(out_dir, "txt.paths"), "w") as fd:
    fd.write("\n".join(s_paths + p_paths))

print("save to :", os.path.join(out_dir, "txt.paths"))

