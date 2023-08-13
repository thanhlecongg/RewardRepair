import datetime
import argparse
import torch
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import loader
from torch.utils.data import DataLoader
import pandas as pd
import csv
import os, gc
import subprocess
import shutil
import numpy as np
SEED=42
LEARNING_RATE = 1e-4
VALID_BATCH_SIZE = 1
MAX_LEN = 384
PATCH_LEN = 76 
TEST_PATH='/repair/dataset/test.csv'

def getBugName(bugid):
    print(bugid)
    bugid=str(bugid).replace(' ','')
    buginfo=''
    startNo=''
    removeNo=''
    filepath=''
    with open(TEST_PATH) as testfile:
        lines = testfile.readlines()
        for l in lines:
            bid=l.split('\t')[0]
            bid=bid.replace(' ','')
            if bid in bugid and bugid in bid:
                buginfo=l.split('\t')[3]
                buginfo=buginfo.replace('\n','').replace('\t','').replace('\r','')
                startNo=l.split('\t')[4]
                removeNo=l.split('\t')[5]
                infos = l.split('\t')
                if len(infos) > 6:
                    filepath=l.split('\t')[6]
                    filepath=filepath.replace('\n','').replace('\t','').replace('\r','')
                else:
                    filepath=''
                break    
    return buginfo,startNo,removeNo,filepath

def current_formatted_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%a %d %b %Y %H:%M:%S %p")

def getGeneratorDataLoader(filepatch,tokenizer,batchsize):
    df = pd.read_csv(filepatch,encoding='latin-1',delimiter='\t').head(1)
    print(df.head(1))
    
    df = df[['bugid','patch','buggy']]

    params = {
        'batch_size': batchsize,
        'shuffle': True,
        'num_workers': 0
        }

    dataset=df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)
    target_set = loader.GeneratorDataset(dataset, tokenizer, MAX_LEN, PATCH_LEN)
    target_loader = DataLoader(target_set, **params)
    return target_loader
        
def test(tokenizer, model, device, loader, output_path, top_n_patches):
    if top_n_patches == -1:
        return_sequences = 200
    else:
        return_sequences = top_n_patches
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            gc.collect()
            torch.cuda.empty_cache()
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            bugid = data['bugid'].to(device, dtype = torch.long)
                        
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=100, 
                num_beams=return_sequences,
                length_penalty=1.0, 
                early_stopping = True,
                num_return_sequences=return_sequences,
                num_beam_groups = 1,
                output_scores=True
                )
           

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            with open(output_path, 'a') as csvfile:
                filewriter = csv.writer(csvfile, delimiter='\t',escapechar=' ',quoting=csv.QUOTE_NONE)
                for i in range(0,return_sequences):
                    predstr=preds[i]
                    predstr=predstr.replace('> =','>=').replace('< =','<=').replace('= =','==').replace('! =','!=')
                    filewriter.writerow([predstr])

                        
def executePerturbation(bugId, src_dir, rootdir="/repair/RewardRepair/"):
    compile_error_flag = True
    project = bugId.split('_')[0]
    bugNo = bugId.split('_')[1]
    exectresult=''
    program_path= src_dir
    print('****************'+program_path+'******************')
   
    #get test result
    cmd = "cd " + program_path + ";"
    cmd += "defects4j info -p "+project +"  -b "+ bugNo
    result=''
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(result)
    failingtest = ''
    faildiag = ''
    if 'Root cause in triggering tests:' in str(result):
        result=str(result).split('Root cause in triggering tests:')[1]
    if '--------' in str(result):
        result=str(result).split('--------')[0]
    
    print(result)
    
    resultLines = str(result).split('\\')
    for l in resultLines:
        if '-' in l and '::' in l and failingtest  in '':
            failingtest = l.split('-')[1]
            failingtest=failingtest.strip()
        if '-->' in l and faildiag  in '':
            faildiag = l.split('-->')[1]
            if '.' in faildiag:
                faildiag_dots = faildiag.split('.')
                if len(faildiag_dots)>2:
                    faildiag=''
                    for i in range(2,len(faildiag_dots)):
                        faildiag+=faildiag_dots[i]
  
    print('==========failingtest======='+failingtest)
    print('==========faildiag======='+faildiag)

    failingTestMethod=failingtest.split('::')[1]
    exectresult = '[FE] ' + faildiag +' '+failingTestMethod
    os.chdir(rootdir)

    return exectresult

def constructTestSample(bugId, indexId, targetfile, repodir, rootdir, startLineNo, buggyLines, patchLines, totalhunk, bno):   
    origTargetFile=targetfile.replace('\r','').replace('\n','')
    className = targetfile.split('/')[-1]
    className = className.replace('.java','').replace('\r','').replace('\n','')
    targetfile=repodir+bugId+'/'+targetfile
    targetfile = targetfile.split('.java')[0]+'.java'
    targetfile=targetfile.replace('\r','').replace('\n','')
    print('targetfile'+targetfile)
    print('startLineNo=========startLineNo====='+startLineNo)
    print('bugId=========bugId====='+bugId)
    print('buggyLines'+buggyLines)
    cxt=''
    metaInfo=''
    diagnosticMsg = executePerturbation(bugId,repodir,rootdir)
    print(diagnosticMsg)


    cmd = 'timeout 200 java -jar /repair/perturbation_model/target/perturbation-0.0.1-SNAPSHOT-jar-with-dependencies.jar '+targetfile+' test-'+startLineNo
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(result)
    result = str(result)
    if '[CLASS]' in result:
        metaInfo = result.split('[CLASS]')[-1]
    if 'startline:' in result:
        cxtStart=result.split('startline:')[1]
        cxtStart=cxtStart.split(' ')[0]
    else:
        cxtStart = int(startLineNo)-10
    if 'endline:' in result:
        cxtEnd=result.split('endline:')[1]
        if '\'' in cxtEnd:
            cxtEnd=cxtEnd.split('\'')[0]
        if '\"' in cxtEnd:
            cxtEnd=cxtEnd.split('\"')[0]
    else:
        cxtEnd=int(startLineNo)+10


    print('meta=========meta====='+metaInfo)
    
    if 'startline' in metaInfo:
        metaInfo = metaInfo.split('startline')[0]
        

        
    if (int(cxtEnd) - int(startLineNo))>10:
        cxtEnd = str(int(startLineNo)+10)
    if (int(startLineNo) - int(cxtStart))>10:
        cxtStart = str(int(startLineNo)-10)       
    cxtStart=str(cxtStart)
    cxtEnd=str(cxtEnd)
      
    print('cxtStart=========cxtStart====='+cxtStart)
    print('cxtEnd=========cxtEnd====='+cxtEnd)

    sample=''
    
    #get context info
    if cxtStart not in '' and cxtEnd not in '':
        with open(targetfile,'r',encoding='latin1') as perturbFile:
            lines = perturbFile.readlines()
            for i in range(0,len(lines)):
                if i > int(cxtStart)-2 and i < int(cxtEnd):
                    l = lines[i]
                    l = l.strip()
                    #remove comments
                    if  l.startswith('/') or l.startswith('*'):
                        l = ' '
                    l = l.replace('  ','').replace('\r','').replace('\n','')
                    l = l.split('// ')[0]
                    if int(bno) > 0:
                        if i == int(startLineNo)-1:
                            l=' [BUGGY] '+l
                        elif i == int(startLineNo)+ int(bno) -1:
                            l= ' [BUGGY] '+l
                    elif int(bno) == 0:
                        if i == int(startLineNo)-1:
                            l=' [BUGGY] [BUGGY] '+l
      
                    cxt+=l+' '

    
    sample+='[BUG] [BUGGY] ' + buggyLines +' '+ diagnosticMsg+' '+' [CONTEXT] ' + cxt + ' [CLASS]  '+ metaInfo

    sample = sample.replace('\r','').replace('\n','').replace('\t','')
    sample = sample.replace('  ',' ')
    print(sample)

    global countindex 
    with open(repodir+'/test.csv','a')  as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t',  escapechar=' ', 
                                quoting=csv.QUOTE_NONE)               
        filewriter.writerow([str(countindex),'[PATCH] '+patchLines,sample,bugId+'_'+className+'_'+totalhunk+'_'+str(int(indexId)+1),startLineNo,str(bno),origTargetFile])
        countindex+=1

def prepare_test_data(bug_id, src_dir, buggy_file, buggy_loc):
    print("====> Preparing test data ...")
    cxt=''
    targetfile = os.path.join(src_dir, buggy_file)
    buggyLines = open(targetfile, "r").readlines()[buggy_loc-1]
    bno = 1
    cmd = 'timeout 200 java -jar /repair/perturbation_model/target/perturbation-0.0.1-SNAPSHOT-jar-with-dependencies.jar ' + targetfile + ' test-' + str(buggy_loc)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(result)
    result = str(result)
    if 'startline:' in result:
        cxtStart=result.split('startline:')[1]
        cxtStart=cxtStart.split(' ')[0]
    else:
        cxtStart = int(buggy_loc)-10
    if 'endline:' in result:
        cxtEnd=result.split('endline:')[1]
        if '\'' in cxtEnd:
            cxtEnd=cxtEnd.split('\'')[0]
        if '\"' in cxtEnd:
            cxtEnd=cxtEnd.split('\"')[0]
    else:
        cxtEnd=int(buggy_loc)+10

    if (int(cxtEnd) - int(buggy_loc))>10:
        cxtEnd = str(int(buggy_loc)+10)
    if (int(buggy_loc) - int(cxtStart))>10:
        cxtStart = str(int(buggy_loc)-10)       
    cxtStart=str(cxtStart)
    cxtEnd=str(cxtEnd)
      
    print('cxtStart=========cxtStart====='+cxtStart)
    print('cxtEnd=========cxtEnd====='+cxtEnd)

    sample=''
    
    #get context info
    if cxtStart not in '' and cxtEnd not in '':
        with open(targetfile,'r',encoding='latin1') as perturbFile:
            lines = perturbFile.readlines()
            for i in range(0,len(lines)):
                if i > int(cxtStart)-2 and i < int(cxtEnd):
                    l = lines[i]
                    l = l.strip()
                    #remove comments
                    if  l.startswith('/') or l.startswith('*'):
                        l = ' '
                    l = l.replace('  ','').replace('\r','').replace('\n','')
                    l = l.split('// ')[0]
                    cxt+=l+' '

    
    sample+='buggy: ' + buggyLines +' '+' context: ' + cxt

    sample = sample.replace('\r','').replace('\n','').replace('\t','')
    sample = sample.replace('  ',' ')
    print("Processed data: " + sample)
    with open(TEST_PATH,'w')  as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t',  escapechar=' ', 
                                quoting=csv.QUOTE_NONE)               
        filewriter.writerow(['bugid','patch','buggy'])
        filewriter.writerow(['0','[PATCH] None', sample])
    
def getFailingTestDiagnostic(failingtest,program_path):
    testclass = failingtest.split("::")[0]

    cmd = "cd " + program_path + ";"
    cmd += "timeout 120 defects4j monitor.test -t "+failingtest
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print('====result===='+str(result))
    if 'failed!' in str(result) :
        result = str(result).split('failed!')[1]
        if testclass in str(result):
            result = str(result).split(testclass)[1]
            if '):' in str(result):
                result = str(result).split('):')[1]
                if '\\' in str(result):
                    result = str(result).split('\\')[0]
    else:
        result =''

    return str(result)
    
def execute(validation_dir, rootdir="/repair/RewardRepair/"):
    compile_error_flag = True

    program_path= validation_dir
    print('****************'+program_path+'******************')
    #get compile result
    cmd = "cd " + program_path + ";"
    cmd += "timeout 90 defects4j compile"
    exectresult='[timeout]'
    symbolVaraible=''
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(result)
    
    # evaluate compilable
    if 'Running ant (compile)' in str(result):
        result = str(result).split("Running ant (compile)")[1]
        result=result.split('\n')
        for i in range(0,len(result)):
            if 'error: ' in result[i]:
                firstError=result[i].split('error: ')[1]
                exectresult=firstError.split('[javac]')[0]
                if '\\' in exectresult:
                    exectresult=exectresult.split('\\')[0]
                print('=======First Error========='+firstError)
                # 'cannot  find  symbol      
                if 'symbol' in firstError and 'cannot' in firstError and 'find' in firstError:       
                    if '[javac]' in firstError:
                        lines = firstError.split('[javac]')
                        for l in lines:
                            if 'symbol:'in l and 'variable' in l:
                                symbolVaraible=l.split('variable')[1]
                                if '\\' in symbolVaraible:
                                    symbolVaraible=symbolVaraible.split('\\')[0]
                                break



                exectresult='[CE] '+exectresult+symbolVaraible
                break
            elif 'OK' in result[i]:               
                exectresult='OK'
                compile_error_flag=False

    # evaluate plausible
    if not compile_error_flag:
        #get test result
        cmd = "cd " + program_path + ";"
        cmd += "timeout 120 defects4j test"
        result=''
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(result)
        if 'Failing tests: 0' in str(result):
            exectresult='[Plausible]'
        elif 'Failing tests' in str(result):
            result=str(result).split('Failing tests:')[1]
            result=str(result).split('-')
            for i in range(1,len(result)):
                failingtest = result[i]
                if '::' not in failingtest and i+1<len(result):
                    failingtest = result[i+1]
                if '\\' in failingtest:
                    failingtest = failingtest.split('\\')[0]
                failingtest=failingtest.strip()

                if '::' in failingtest:
                    failingTestMethod=failingtest.split('::')[1]
                    faildiag = getFailingTestDiagnostic(failingtest,program_path)
                    exectresult = '[FE] ' + faildiag +' '+failingTestMethod
                else:
                    exectresult = '[FE] '
                break
   
    os.chdir(rootdir)

    return exectresult

def executePatch(validation_dir, patch, origin_buggy_file, buggy_file, buggy_loc):
    #keep a copy of the buggy file
    newStr=''
    startNo = buggy_loc
    endNo= buggy_loc + 1
    
    buggy_file = os.path.join(validation_dir, buggy_file)
    with open(buggy_file, 'r') as of:
        lines=of.readlines()
        for i in range(0,len(lines)):
            l=lines[i]
            if i+1 < int(startNo):
                newStr +=l 
            if i+1 == int(startNo):
                newStr += patch + '\n'
            if i+1 >= endNo:
                newStr+=l
    
    with open(buggy_file,'w') as wof:
        wof.write(newStr)

    exeresult = execute(validation_dir)
        
    os.system("cp -r {} {}".format(origin_buggy_file, buggy_file))

    return exeresult

def validate(bug_id, src_dir, buggy_file, buggy_loc, output_dir="./"):
    print("====> Validating generated patches")
    validation_dir = '/tmp/' + bug_id
    subprocess.run('rm -rf ' + validation_dir, shell=True)
    shutil.copytree(src_dir, validation_dir)
    patchFromPath=os.path.join(output_dir, "raw_results.csv")
    patchToPath=os.path.join(output_dir, "validated_patches.csv")
    patchFolder=os.path.join(output_dir, "patches")
    os.makedirs(patchFolder, exist_ok=True)
    if os.path.exists(patchToPath):
        os.remove(patchToPath)
    origin_buggy_file = os.path.join(src_dir, buggy_file)
    buggyLines = open(origin_buggy_file, "r").readlines()[buggy_loc-1]

    with open(patchFromPath,'r') as patchFile:
        patches = patchFile.readlines()
        for idx, patch in enumerate(patches):
            print(patch)
            exeresult = executePatch(validation_dir, patch, origin_buggy_file, buggy_file, buggy_loc)
            with open(patchToPath,'a') as targetFile:
                targetFile.write(exeresult+'\t'+str(idx)+'\t'+patch)
            
            if exeresult == "[Plausible]":
                diff_path = os.path.join(patchFolder, f"{idx}.diff")
                with open(diff_path, "w") as f:
                    f.write('- ' + buggyLines.strip() + "\n")
                    f.write('+ ' + patch.strip())
        
def repair(output_path, top_n_patches):
    LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    SUMMARY_LEN = 64 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained('/repair/model/RewardRepair')
    tokenizer.add_tokens(['{', '}','<','^'])

    model = T5ForConditionalGeneration.from_pretrained('/repair/model/RewardRepair')
    model = model.to(device)


    test_df = pd.read_csv(TEST_PATH,encoding='latin-1',delimiter='\t')
    print(test_df.head())
    test_df = test_df[['bugid','buggy','patch']]
    print(test_df.head())

    
    test_dataset=test_df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)


    print("TEST Dataset: {}".format(test_dataset.shape))



    test_set = loader.GeneratorDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    
    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 2
        }

    test_loader = DataLoader(test_set, **test_params)  

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(0,1):
        test(tokenizer, model, device, test_loader, output_path, top_n_patches)  
                  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--bug_id', type=str, default='Chart_1')
    parser.add_argument('--src_dir', type=str, default='test_prj')
    parser.add_argument('--buggy_file', type=str, default='source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java')
    parser.add_argument('--buggy_loc', type=int, default=1797)
    parser.add_argument('--output_folder', type=str, default='/output/')
    parser.add_argument('--top_n_patches', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    print("Start Time: " + current_formatted_time())
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    
    global device 
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.task == "repair":
        prepare_test_data(args.bug_id, args.src_dir, args.buggy_file, args.buggy_loc)
        warnings.filterwarnings('ignore')
        print(torch.__version__)
        gc.collect()
        torch.cuda.empty_cache()
        output_path = os.path.join(args.output_folder, "raw_results.csv")
        if os.path.exists(output_path):
            os.remove(output_path)
        repair(output_path, args.top_n_patches)
    elif args.task == "validate":
        validate(args.bug_id, args.src_dir, args.buggy_file, args.buggy_loc, args.output_folder)
    print("End Time: " + current_formatted_time())
