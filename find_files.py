import os,sys
def find_func_files(ROOT_PATH='path-to-dataset'):
    files=[]

    for root,d_names,f_names in os.walk(ROOT_PATH):
        for f_name in f_names:
            if f_name.endswith("_task-rest_bold.nii.gz"):
                files.append(ROOT_PATH+f_name.split('_')[0]+'/func/'+f_name)

    return files

