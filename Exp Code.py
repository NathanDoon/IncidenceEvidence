'''
!!! This Script assumes you have already ran / loaded the "IEFuncs" script. !!!!
#Step 1
Find Corups
'''
from scipy.stats import zscore; import pandas as pd
corpus = "Your File Path here to 'TextFiles' Folder]"


'''
#Step 2 -- Ensure Following Matrices can be constructed:
    200/100/50 MFW
    200/100/50 MFW + Oras                          
    200/100/50 MFW + Timberlake Strict                     
    200/100/50 MFW + Timberlake Total
    200/100/50 MFW + Oras + Timberlake Strict
    200/100/50 MFW + Oras + Timberlake Total
    
    Oras
    Timberlake Strict
    Timberlake Total
    Oras + Timberlake Strict
    Oras + Timberlake Total
    (All as Words & Z-scores)
'''
DataFrameTestSet = []
MFW_Range = [50, 100, 200]

for value in MFW_Range:
    MFW = DFM(corpus, MFW = value, Culling = 25)                                                                                                      #Create Matrix of Word Counts
    MFW = MFW.apply(zscore, ddof = 1)                                                                                                               #Turn Word Counts into Z-scores
    
    #ORAS
    Plain_Oras = pd.read_csv("[Your File Path here]/Oras Raw.csv", index_col=[0])                
    Plain_Oras.index = [entry + ".txt" for entry in Plain_Oras.index]                                                                               #Used to make Oras indexes same as Word Matrix
    Z_Oras = Plain_Oras.apply(zscore, ddof = 1)                                                                                                     #Turn Oras Counts into Z-Scores
    
    
    #TIMBERLAKE Permutations belows
    Plain_Timberlake = pd.read_csv("Your File Path here]/Timberlake Raw.csv", index_col =[0])
    Plain_Timberlake.index = [entry + ".txt" for entry in Plain_Timberlake.index]
    
    Plain_Total_Timberlake = Plain_Timberlake.drop(axis = 1, columns = "Strict") #Also works with "Strict" as drop argument. Used to keep only 1 Timberlake
    Plain_Strict_Timberlake = Plain_Timberlake.drop(axis = 1, columns = "Total")
    
    Z_Total_Timberlake = Plain_Total_Timberlake.apply(zscore, ddof = 1) #OPTIONAL: Applies Z-scores to Percentage values from Oras
    Z_Strict_Timberlake = Plain_Strict_Timberlake.apply(zscore, ddof = 1)


    MFW_Oras_Z = pd.concat([MFW, Z_Oras], axis = 1)
    MFW_Oras_Plain = pd.concat([MFW, Plain_Oras], axis = 1)
    MFW_Timberlake_Strict_Plain = pd.concat([MFW, Plain_Strict_Timberlake], axis = 1)
    MFW_Timberlake_Strict_Z = pd.concat([MFW, Z_Strict_Timberlake], axis = 1)
    MFW_Timberlake_Total_Plain = pd.concat([MFW, Plain_Total_Timberlake], axis = 1)
    MFW_Timberlake_Total_Z = pd.concat([MFW, Z_Total_Timberlake], axis = 1)
    
    MFW_Oras_Timberlake_Strict_Z = pd.concat([MFW, Z_Strict_Timberlake, Z_Oras], axis = 1)
    MFW_Oras_Timberlake_Total_Z = pd.concat([MFW, Z_Total_Timberlake, Z_Oras], axis = 1)
    
    DataFrameTestSet.append(MFW)
    DataFrameTestSet.append(MFW_Oras_Z)
    DataFrameTestSet.append(MFW_Oras_Plain)
    DataFrameTestSet.append(MFW_Timberlake_Strict_Plain)
    DataFrameTestSet.append(MFW_Timberlake_Strict_Z)
    DataFrameTestSet.append(MFW_Timberlake_Total_Plain)
    DataFrameTestSet.append(MFW_Timberlake_Total_Z)
    DataFrameTestSet.append(MFW_Oras_Timberlake_Strict_Z)
    DataFrameTestSet.append(MFW_Oras_Timberlake_Total_Z)

Timberlake_Oras_Only = pd.concat([Plain_Total_Timberlake, Plain_Oras], axis = 1)
Timberlake_Oras_Z = pd.concat([Z_Total_Timberlake, Z_Oras], axis = 1)

DataFrameTestSet.append(Plain_Oras)
DataFrameTestSet.append(Z_Oras)
DataFrameTestSet.append(Plain_Strict_Timberlake)
DataFrameTestSet.append(Plain_Total_Timberlake)
DataFrameTestSet.append(Z_Strict_Timberlake)
DataFrameTestSet.append(Z_Total_Timberlake)
DataFrameTestSet.append(Timberlake_Oras_Only)
DataFrameTestSet.append(Timberlake_Oras_Z)




#######################################################
'''
#Step 3
Use Cartesian Function to Create Training Data Sets 

&& 

#Step 4
Fill Cartesian Lists with Relevant Plays for Test Classifications

'''
ToClassify = []
for df in DataFrameTestSet:
    ComboSet = CartesianProduct(df, TPA = 1)


    Texts = [entry for entry in list(df.index)]
    Original = list(df.index) 
    
    for i in range (0, len(ComboSet[0])):
        ANON_Indexes = []
        ANON_Texts = [entry for entry in Texts if entry not in ComboSet[1][i]]
        
        
        for text in ANON_Texts:
            Inds = [i for i, s in enumerate(Original) if text in s]
            ANON_Indexes.append(Inds)
        
        ANON_Texts = ["ANON_" + entry.split("_", 1)[1] for entry in ANON_Texts]
        ComboSet[2][i] = ANON_Indexes
        ComboSet[3][i] = ANON_Texts
    ToClassify.append(ComboSet)
#####################################################################################################################################################################


'''
#Step 5
Run KNN Classifications on Cartesian Test Set

&&&

#Step 6
Run SVM Classifications on Cartesian Test Set

'''
#Step 5 & 6 Loop --- Classify Using KNN & SVM
for k in range (0, len(DataFrameTestSet)):
    DF_to_use = DataFrameTestSet[k]
    ComboSet = ToClassify[k]
    
    ######STEP 5 --- KNN CLASSIFICATION 
    Results = []
    for i in range (0, len(ComboSet[0])):
        #Make Classes just Author name, so that it doesn't think each chunk is its own class
        Train_Classes = [entry.split("_", 1)[0] for entry in ComboSet[1][i]]
        Train_Data = DF_to_use.iloc[list(ComboSet[0][i])]
        Test_Data = DF_to_use.iloc[flatten(ComboSet[2][i])]
        Test_Titles = ComboSet[3][i]
        Res = KNN_Classify(TrainingData = Train_Data, TrainingClasses = Train_Classes, 
                           TestData = Test_Data, TestTitles = Test_Titles, 
                           Distance = "cosine", Neighbours = 1)
        
        ToAppend = tuple(Res)
        Results.append(ToAppend)
    
    
    x = pd.DataFrame(Results)
    x.to_csv("[Your File Path here for Classifications Output]" + "KNN-" + str(k) + ".csv")
    
    
        
    ######STEP 6 --- SVM CLASSIFICATION 
    Results = []
    for i in range (0, len(ComboSet[0])):
        #SVM Classifer bases class name off of index, not class list. Make index just author name so each chunk not considered own class.
        Train_Classes = [entry.split("_", 1)[0] for entry in ComboSet[1][i]]
        TD =  DF_to_use.iloc[list(ComboSet[0][i])]
        Test_Data = DF_to_use.iloc[flatten(ComboSet[2][i])]
        Test_Titles = ComboSet[3][i]
        
        Res = SVM_Classify(TrainingData = TD, TrainingClasses = Train_Classes, 
                           TestData = Test_Data, TestTitles = Test_Titles,
                           Kernel = "linear", C = 1.0)

        ToAppend = tuple(Res)
        Results.append(ToAppend)
        
    x = pd.DataFrame(Results)
    x.to_csv("[Your File Path here for Classifications Output]" + "SVM-" + str(k) + ".csv")










'''
#Step 7
Summarize Table
'''
import os; from os import listdir; import pandas as pd; import string; import re
path = "[Your File Path here for folder with Classification Output CSV's']"

#Get Every CSV from the assigned File Path
fileSet = [entry for entry in listdir(path) if ".csv" in entry]

Assignments = {} #Records every time a particular candidate wins for a particular DF
TotalExps = {}  #Every time an author is included in a set of classifications, we +=1 this, so that we can divide their total wins by this dictionary's value for getting a % later




#Begin Summarizing Loop
for entry in fileSet:
    y = pd.read_csv(path + "/" + entry)
    y.drop("Unnamed: 0", axis = 1, inplace = True)
    
    z = list(y.itertuples(index=False, name=None))
    
    test = [list(entry) for entry in z]
    test = flatten(z)



    for i in range (0, len(test)):
        AssignedAuthor = test[i].split("', '")[-1]
        AssignedAuthor = AssignedAuthor.rsplit("')")[0]
        
        CorrectAuthor = test[i].split("_(")[-1]  
        CorrectAuthor = CorrectAuthor.rsplit(").txt")[0]
            
        DF_Code = entry.split("-")[-1]
        DF_Code = DF_Code.rsplit(".csv")[0]    
        Key = DF_Code + " Success"
        
        if (AssignedAuthor == CorrectAuthor):
            UpdateDict(Assignments, Key)
        
        Key = DF_Code + " Total"
        UpdateDict(TotalExps, Key)
        



















