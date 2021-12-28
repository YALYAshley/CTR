import os
import shutil

new_folder1 = 'left'
path1 = '左'

new_folder2 = 'right'
path2 = '右'

new_folder3 = 'center'
path3 = '中'

start_num = 721                           #注意修改
name_format = '20211228_1528'
images1 = os.listdir(path1)
images2 = os.listdir(path2)
images3 = os.listdir(path3)

os.chdir(path1)
for image in images1:
    id = int(image.strip('.jpg'))
    im_id = id-start_num
    im_id = str(im_id)
    im_id = im_id.zfill(3)
    im_name = name_format+im_id+'.jpg'
    os.rename(image,im_name)
    for i in range(1,6):
        i = str(i).zfill(2)
        shutil.copyfile(im_name,'../'+new_folder1+'/'+im_name.strip('.jpg')+'_'+i+'.jpg')


os.chdir('../'+path2)
for image in images2:
    id = int(image.strip('.jpg'))
    im_id = id-start_num
    im_id = str(im_id)
    im_id = im_id.zfill(3)
    im_name = name_format+im_id+'.jpg'
    os.rename(image,im_name)
    for i in range(1,6):
        i = str(i).zfill(2)
        shutil.copyfile(im_name,'../'+new_folder2+'/'+im_name.strip('.jpg')+'_'+i+'.jpg')

os.chdir('../'+path3)
for image in images3:
    id = int(image.strip('.jpg'))
    im_id = id-start_num
    im_id = str(im_id)
    im_id = im_id.zfill(3)
    im_name = name_format+im_id+'.jpg'
    os.rename(image,im_name)
    for i in range(1,6):
        i = str(i).zfill(2)
        shutil.copyfile(im_name,'../'+new_folder3+'/'+im_name.strip('.jpg')+'_'+i+'.jpg')