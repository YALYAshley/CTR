import json
import os

path = '1'
for pose in os.listdir(path):
    print(pose)
    for pscp in os.listdir(os.path.join(path,pose)):
        jsfile = os.listdir(path+'/'+pose+'/'+pscp)
        with open(path+'/'+pose+'/'+pscp+'/'+jsfile[0],'r') as f:
            js = json.load(f)
            for k,v in js.items():
                k = k.strip('.jpg')
                print(k)
                with open(path+'/'+pose+'/'+pscp+'/'+k+'.json','a') as nf:
                    dict = {k:v}
                    json.dump(dict,nf)