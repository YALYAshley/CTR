import json
import os

path = '/home/mn/8T/code/MVSTHGNN/data/image/'
dirpath = '1'
poses = os.listdir(dirpath)
for pose in poses:
    print(pose)
    for pspc in os.listdir(os.path.join(dirpath,pose)):
        print(pspc)
        if pose=='turnright' and pspc=='center':
            print(os.listdir(os.path.join(dirpath,pose,pspc)))
        for jsfile in os.listdir(os.path.join(dirpath,pose,pspc)):
            

            if jsfile.endswith('.json'):
                with open(os.path.join(dirpath,pose,pspc,jsfile),'r') as f:
                    js = json.load(f)
                    key = js.keys()
                    #print(len(list(js.values())))
                    dict = {jsfile.strip('.json')+'.jpg':list(js.values())[0]}
                    bodies = dict[jsfile.strip('.json')+'.jpg']['bodies']
                
                    #print(type(bodies))
                    for id,body in enumerate(bodies):
                        #print(id)
                        dict[jsfile.strip('.json')+'.jpg']['bodies'][id]['im_name'] =path+pose+'/'+pspc+'/'+jsfile.strip('.json')+'.jpg'
                        # body['im_name'] = jsfile.strip('.json')+'.jpg'
                    with open(os.path.join('new_data',pose,pspc,jsfile),'a') as nf:
                        json.dump(dict,nf)