#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[16]:


import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time


# # Reading file

# In[17]:


f = open('Brown_train.txt','r')
filedata = f.read()
#print(filedata)
#print(type(filedata))


# In[18]:


##Reference : https://github.com/shikhinmehrotra/NLP-POS-tagging-using-HMMs-and-Viterbi-heuristic/blob/master/NLP-POS%20tagging%20using%20HMMs%20and%20Viterbi%20heuristic.ipynb


# # Splitting into sentences

# In[149]:


file_sentences = filedata.split("\n")
length_sentences = []
for i in range(0,len(file_sentences)):
    length_sentences.append(len(file_sentences[i]))

sum = 0
for i in length_sentences:
    sum += i
avg_length_of_sentence = sum/(len(file_sentences))

max_length_of_sentence = max(length_sentences)
min_length_of_sentence = min(length_sentences)
print('The number of sentences is {}'.format(len(file_sentences)))
print('maximum length of sentence is {}'.format(max_length_of_sentence))
print('minimum length of sentence is {}'.format(min_length_of_sentence))
print('average length of sentence is {}'.format(avg_length_of_sentence))


# # Splitting Sentences into words and tags

# In[20]:


words_and_tags = []
words = []
tags = []
wat = []
full_sent_tags = []
for i in range(0,len(file_sentences)):
    dummy = file_sentences[i].split(" ")
    dummy = dummy[:-1]
    dummy_wat = []
    sent_tags = []
    for j in range(0,len(dummy)):
        words_and_tags.append(dummy[j])
        dummy1 = dummy[j].split("_")
        words.append(dummy1[0])            
        tags.append(dummy1[1])
        dummy_wat.append(tuple(dummy1))
        sent_tags.append(tuple(dummy))
    full_sent_tags.append(list(sent_tags))
    wat.append(dummy_wat)


# In[21]:


## words_and_tags is list of word-tags
## words is the list of words
## tags is the list of tags
## wat is the list of list of tuples containing word and tag as a tuple


# In[22]:


print('length of words_and_tags is : {}'.format(len(words_and_tags)))
print('length of words is          : {}'.format(len(words)))
print('length of tags is           : {}'.format(len(tags)))
print('length of wat is            : {}'.format(len(wat)))


# In[23]:


# Getting list of tagged words
tagged_words = [tup for sent in wat for tup in sent]
len(tagged_words)


# In[24]:


# vocabulary : Unique words
Vocab = set(words)
print('Number of unique words in corpus is : {}'.format(len(Vocab)))


# In[25]:


Tag_set = set(tags)
print('Number of unique tags in corpus is : {}'.format(len(tags)))


# In[26]:


start_time = time.time()
Tag_count = {}

for tag in list(Tag_set):
    Tag_count[tag] = 0
    
for pair in tagged_words:
        Tag_count[pair[1]]+=1
end_time = time.time()
difference = end_time - start_time
print('The time taken to calculate tag count is : {}'.format(difference))


# In[27]:


sum = 0
for key in Tag_count:
    print('For tag {},               the count of words is {}'.format(key,Tag_count[key]))
    sum += Tag_count[key]


# In[28]:


print(sum)


# In[29]:


print(len(Tag_count))


# In[30]:


## Reference for Emission and Transition Probabilities
## https://youtu.be/68hmUltbPnw

## Reference taken is Textbook posted in google classroom


# # Emission Probabilities

# In[31]:


## Emissionn probabilities : Probabilities of a word given tag is stored in a numpy array
## The size of the martrix will be (number_of_unique_tags * number_of_unique_words)
## In this case, the size of emission probabilities matrix is going to be (Tag_set * Vocab)


# In[32]:


start_time = time.time()
EmissionProbs = {}  #Empty dictionary to store emission probabilities

WordToTag = {}   #Empty dictionary to store the count of word given tag

for word in list(Vocab):
    WordToTag[word] = {key:0 for key in Tag_set}

for pair in tagged_words:
    for word in pair:
        WordToTag[pair[0]][pair[1]]+=1
        
for word in list(Vocab):
    EmissionProbs[word] = {key:0 for key in Tag_set}
    
for x in WordToTag:
    for reqTag in WordToTag[x]:
        if WordToTag[x][reqTag]==0 or Tag_count[reqTag]==0:
            EmissionProbs[x][reqTag]=0
        else:
            EmissionProbs[x][reqTag]=(WordToTag[x][reqTag])/Tag_count[reqTag]

end_time = time.time()
difference = end_time - start_time
print('Start time : {}'.format(start_time))
print('End time   : {}'.format(end_time))
print('The time taken to calculate emission probs is : {} seconds'.format(difference))


# In[33]:


for key in EmissionProbs['The']:
    print('{}:{}'.format(key,EmissionProbs['The'][key]))


# # Transition Probabilities

# In[34]:


## Transition Probabilities : Probability of a tag t2 given that the previous tag is t1
## Transition Probability matrix size will be (no_of_unique_tags * no_of_unique_tags)
## In this case, the size is (len(Tag_set) * len(Tag_set))


# In[35]:


start_time = time.time()
TransitionProbs = {}  #Empty dictionary to store transition probabilities

Tag2ToTag1_count = {}

for tag in list(Tag_set):
    Tag2ToTag1_count[tag] = {key:0 for key in Tag_set}
    
#print('Fulton_NP-TL' in Tag_set)
for sent in wat:
    for x in range(len(sent)-1):
        Tag2ToTag1_count[sent[x][1]][sent[x+1][1]]+=1

for tag in list(Tag_set):
    TransitionProbs[tag] = {key:0 for key in Tag_set}
    
for x in Tag2ToTag1_count:
    for reqTag in Tag2ToTag1_count[x]:
        if Tag2ToTag1_count[x][reqTag]==0 or Tag_count[reqTag]==0:
            TransitionProbs[x][reqTag]=(1)/len(Tag_set)
        else:
            TransitionProbs[x][reqTag]=(Tag2ToTag1_count[x][reqTag]+1)/(Tag_count[reqTag]+len(Tag_set))

end_time = time.time()
difference = end_time - start_time
print('Start time : {}'.format(start_time))
print('End time   : {}'.format(end_time))
print('The time taken to calculate emission probs is : {} seconds'.format(difference))


# In[36]:


print(TransitionProbs['.']['NN'])


# # Viterbi Algorithm

# In[37]:


def Viterbi(words,tagged_words):
    #list for predicted tags for words given
    predicted_tags = []
    T = list(set([pair[1] for pair in tagged_words]))
    
    for i in range(len(words)):
        #Probability of every tag for particular word
        dummy_prob = []
        for t in T:
            #Transition Pobability
            if i==0:
                tp = TransitionProbs['.'][t]
            else:
                tp = TransitionProbs[predicted_tags[-1]][t]
            
            
            #Emission Probability
            ep = EmissionProbs[words[i]][t]
            
            #State Probability
            state_prob = tp*ep
            
            dummy_prob.append(state_prob)
            
        max_prob = max(dummy_prob)
        #Getting a tag with max prob
        max_state = T[dummy_prob.index(max_prob)]
        predicted_tags.append(max_state)
    return list(predicted_tags)


# In[38]:


print(Viterbi(['The'],tagged_words))


# # Function for computing Confusion Matrix

# In[39]:


##Reference : https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python


# In[40]:


## This function return numpy array as a resultant confusion matrix
## 'true' indicates the ground truth tags
## 'pred' indicates the predicted tags


# In[41]:


def compute_confusion_matrix(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    
    K = len(Tag_set)+1 # Number of classes 
    result = np.zeros((K, K))
    Tag_list = list(Tag_set)

    for i in range(len(true)):
        result[Tag_list.index(true[i])][Tag_list.index(pred[i])] += 1
        
    for i in range(len(Tag_set)-1):
        sum =0
        for j in range(len(Tag_set)-1):
            sum += result[i][j]
        result[i][len(Tag_set)]=sum
    
    for j in range(len(Tag_set)-1):
        sum = 0
        for i in range(len(Tag_set)-1):
            sum+=result[i][j]
        result[len(Tag_set)][j]=sum
    return result


# # Function for Accuracy

# In[42]:


def compute_accuracy(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    count=0
    for i in range(len(true)):
        if(true[i]==pred[i]):
            count+=1
    print('correct predictions are : {}'.format(count))
    return count/len(true)


# # Function for Precision,Recall and F1-Score

# In[43]:


##Reference : https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal


# In[44]:


## Reference : https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/


# In[45]:


def compute_vals(confusion_matrix):
    count_of_tags = len(list(Tag_set))
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    dict1 = {}
    for i in range(len(Tag_set)):
        TP=confusion_matrix[i][i]
        FP=confusion_matrix[i][count_of_tags]-TP
        FN=confusion_matrix[count_of_tags][i]-TP
        #print(tags_names[i],TP,FP,FN)
        p=0
        r=0
        f=0
        if TP+FP!=0:
            p=TP/(TP+FP)
        if TP+FN!=0:
            r=TP/(TP+FN)
        if p+r!=0:
            f=(2*p*r)/(p+r)
        dict2 = {'precision':p,'recall':r,'f1_score':f}
        dict1.update({list(Tag_set)[i]:dict2})
        total_precision+=p
        total_recall+=r
        total_f1_score+=f
    total_precision = total_precision/len(list(Tag_set))
    total_recall = total_recall/len(list(Tag_set))
    total_f1_score = total_f1_score/len(list(Tag_set))
    return dict1,total_precision,total_recall,total_f1_score


# # Function for Wrongly Predicted Words

# In[162]:


def compute_wrongs(words,true,pred):
    true = np.array(true)
    pred = np.array(pred)
    words = np.array(words)
    ans = []
    for i in range(len(true)):
        if(true[i]!=pred[i]):
            d = []
            d.append(words[i])
            d.append(true[i])
            d.append(pred[i])
            ans.append(list(d))
    return ans


# In[ ]:





# # 3-fold Cross Validation

# # Dividing into Three folds

# In[46]:


print(len(wat))


# In[47]:


55146/3


# In[48]:


first_wat = wat[0:18382]
second_wat = wat[18382:36764]
third_wat = wat[36764:]


# In[49]:


print(len(first_wat))
print(len(second_wat))
print(len(third_wat))


# In[50]:


print(type(first_wat))


# # Considering first fold as a validation set

# In[51]:


validation1 = first_wat
training1 = second_wat+third_wat

print(len(validation1))
print(len(training1))


# In[52]:


validation_words_1 = []
validation_ground_truth_1 = []
for sent in validation1:
    for pair in sent:
        validation_words_1.append(pair[0])
        validation_ground_truth_1.append(pair[1])


# In[53]:


start_time = time.time()
validation_predicted_1 = Viterbi(validation_words_1,tagged_words)
end_time = time.time()
difference = end_time - start_time
print('The time taken for predicting tags on validation set 1 is : {}'.format(difference))


# In[54]:


print('Number of words for fold 1 is           {}'.format(len(validation_words_1)))
print('Number of actual tags for fold 1 is     {}'.format(len(validation_ground_truth_1)))
print('Number of predicted tags for fold 1 is  {}'.format(len(validation_predicted_1)))


# In[163]:


wrong_words_fold1 = compute_wrongs(validation_words_1,validation_ground_truth_1,validation_predicted_1)


# In[164]:


print(len(wrong_words_fold1))


# In[165]:


print('The Wrongly predited words in fold 1 are:-')
for i in wrong_words_fold1:
    print(i)


# In[55]:


# for i in range(10):
#     print('{} and {}'.format(validation_ground_truth_1[i],validation_predicted_1[i]))


# In[56]:


start_time = time.time()
confusion_matrix_1 = compute_confusion_matrix(validation_ground_truth_1,validation_predicted_1)
end_time = time.time()
difference = end_time - start_time
print('The time taken for computing confusion matrix for fold 1 is : {}'.format(difference))


# In[57]:


print(confusion_matrix_1)


# In[58]:


start_time = time.time()
accuracy_1 = compute_accuracy(validation_ground_truth_1,validation_predicted_1)
accuracy_1 *= 100
end_time = time.time()
diff = end_time - start_time
print('The time taken for computing accuracy for fold 1 is : {}'.format(diff))
print('THE ACCURACY FOR FOLD 1 is : {}'.format(accuracy_1))


# In[59]:


tag_details_fold1,precision_fold1,recall_fold1,f1_score_fold1=compute_vals(confusion_matrix_1)


# In[60]:


print('Precision for fold-1 is : {}'.format(precision_fold1))
print('Recall for fold-1 is    : {}'.format(recall_fold1))
print('F1-score for fold-1 is  : {}'.format(f1_score_fold1))


# In[61]:


for key in tag_details_fold1:
    print(key,'->',tag_details_fold1[key])


# # Considering Second fold as validation set

# In[62]:


validation2 = second_wat
training2 = first_wat+third_wat

print(len(validation2))
print(len(training2))


# In[63]:


validation_words_2 = []
validation_ground_truth_2 = []
for sent in validation2:
    for pair in sent:
        validation_words_2.append(pair[0])
        validation_ground_truth_2.append(pair[1])


# In[64]:


start_time = time.time()
validation_predicted_2 = Viterbi(validation_words_2,tagged_words)
end_time = time.time()
difference = end_time - start_time
print('The time taken for predicting tags on validation set 2 is : {}'.format(difference))


# In[65]:


print('Number of words for fold 2 is           {}'.format(len(validation_words_2)))
print('Number of actual tags for fold 2 is     {}'.format(len(validation_ground_truth_2)))
print('Number of predicted tags for fold 2 is  {}'.format(len(validation_predicted_2)))


# In[166]:


wrong_words_fold2 = compute_wrongs(validation_words_2,validation_ground_truth_2,validation_predicted_2)


# In[167]:


print(len(wrong_words_fold2))


# In[168]:


print('The Wrongly predited words in fold 2 are:-')
for i in wrong_words_fold2:
    print(i)


# In[66]:


# for i in range(10):
#     print('{} and {}'.format(validation_ground_truth_2[i],validation_predicted_2[i]))


# In[67]:


start_time = time.time()
confusion_matrix_2 = compute_confusion_matrix(validation_ground_truth_2,validation_predicted_2)
end_time = time.time()
difference = end_time - start_time
print('The time taken for computing confusion matrix for fold 2 is : {}'.format(difference))


# In[68]:


print(confusion_matrix_2)


# In[69]:


start_time = time.time()
accuracy_2 = compute_accuracy(validation_ground_truth_2,validation_predicted_2)
accuracy_2 *= 100
end_time = time.time()
diff = end_time - start_time
print('The time taken for computing accuracy for fold 2 is : {}'.format(diff))
print('THE ACCURACY FOR FOLD 2 is : {}'.format(accuracy_2))


# In[70]:


tag_details_fold2,precision_fold2,recall_fold2,f1_score_fold2=compute_vals(confusion_matrix_2)


# In[71]:


print('Precision for fold-2 is : {}'.format(precision_fold2))
print('Recall for fold-2 is    : {}'.format(recall_fold2))
print('F1-score for fold-2 is  : {}'.format(f1_score_fold2))


# In[72]:


for key in tag_details_fold2:
    print(key,'->',tag_details_fold2[key])


# # Considering third fold as a validation set

# In[73]:


validation3 = third_wat
training3 = first_wat+second_wat

print(len(validation3))
print(len(training3))


# In[74]:


validation_words_3 = []
validation_ground_truth_3 = []
for sent in validation3:
    for pair in sent:
        validation_words_3.append(pair[0])
        validation_ground_truth_3.append(pair[1])


# In[75]:


start_time = time.time()
validation_predicted_3 = Viterbi(validation_words_3,tagged_words)
end_time = time.time()
difference = end_time - start_time
print('The time taken for predicting tags on validation set 3 is : {}'.format(difference))


# In[76]:


print('Number of words for fold 3 is           {}'.format(len(validation_words_3)))
print('Number of actual tags for fold 3 is     {}'.format(len(validation_ground_truth_3)))
print('Number of predicted tags for fold 3 is  {}'.format(len(validation_predicted_3)))


# In[169]:


wrong_words_fold3 = compute_wrongs(validation_words_3,validation_ground_truth_3,validation_predicted_3)


# In[170]:


print(len(wrong_words_fold3))


# In[171]:


print('The Wrongly predited words in fold 3 are:-')
for i in wrong_words_fold3:
    print(i)


# In[77]:


# for i in range(10):
#     print('{} and {}'.format(validation_ground_truth_2[i],validation_predicted_2[i]))


# In[78]:


start_time = time.time()
confusion_matrix_3 = compute_confusion_matrix(validation_ground_truth_3,validation_predicted_3)
end_time = time.time()
difference = end_time - start_time
print('The time taken for computing confusion matrix for fold 3 is : {}'.format(difference))


# In[79]:


print(confusion_matrix_3)


# In[80]:


start_time = time.time()
accuracy_3 = compute_accuracy(validation_ground_truth_3,validation_predicted_3)
accuracy_3 *= 100
end_time = time.time()
diff = end_time - start_time
print('The time taken for computing accuracy for fold 3 is : {}'.format(diff))
print('THE ACCURACY FOR FOLD 3 is : {}'.format(accuracy_3))


# In[81]:


tag_details_fold3,precision_fold3,recall_fold3,f1_score_fold3=compute_vals(confusion_matrix_3)


# In[82]:


print('Precision for fold-3 is : {}'.format(precision_fold3))
print('Recall for fold-3 is    : {}'.format(recall_fold3))
print('F1-score for fold-3 is  : {}'.format(f1_score_fold3))


# In[83]:


for key in tag_details_fold3:
    print(key,'->',tag_details_fold3[key])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Trigrams

# # Transition Probabilities for Trigrams

# In[84]:


## Transition Probabilities : Probability of a tag t3 given that the previous tags are t2 and t1


# In[107]:


Tag_list = list(Tag_set)
newTag_list = []
for i in range(len(Tag_list)):
    for j in range(len(Tag_list)):
        if Tag_list[i] == Tag_list[j]:
            newTag_list.append(Tag_list[i]+'!'+Tag_list[j])
        else:
            newTag_list.append(Tag_list[i]+'!'+Tag_list[j])
            newTag_list.append(Tag_list[j]+'!'+Tag_list[i])    


# In[108]:


print(len(newTag_list))


# In[116]:


start_time = time.time()
TransitionProbs_Trigrams = {}  #Empty dictionary to store transition probabilities

Tag3GivenTag2Tag1_count = {}

for tag in list(newTag_list):
    Tag3GivenTag2Tag1_count[tag] = {key:0 for key in Tag_list}
    
for sent in wat:
    for x in range(len(sent)-2):
        dummy=sent[x+1][1]+'!'+sent[x+2][1]
        Tag3GivenTag2Tag1_count[dummy][sent[x][1]]+=1
        
for tag in list(newTag_list):
    TransitionProbs_Trigrams[tag] = {key:0 for key in Tag_list}
    
for x in Tag3GivenTag2Tag1_count:
    for reqTag in Tag3GivenTag2Tag1_count[x]:
        seperate_tags = x.split('!')
        d1=seperate_tags[0]
        d2=seperate_tags[1]
        if Tag3GivenTag2Tag1_count[x][reqTag]==0 or Tag_count[reqTag]==0:
            TransitionProbs_Trigrams[x][reqTag]=(1)/len(Tag_set)
        else:
            TransitionProbs_Trigrams[x][reqTag]=(Tag3GivenTag2Tag1_count[x][reqTag]+1)/(TransitionProbs[d1][d2]+len(Tag_set))
        #print('x is {} reqTag is {} prob is {}'.format(x,reqTag,TransitionProbs_Trigrams[x][reqTag]))

    
end_time = time.time()
diff = end_time - start_time
print('Time taken for calculation for Tranition Probabilities for Trigrams is {}'.format(diff))


# # Viterbi for Trigrams

# In[133]:


def Viterbi_Trigrams(words,tagged_words):
    #list for predicted tags for words given
    predicted_tags = []
    T = list(set([pair[1] for pair in tagged_words]))
    
    for i in range(len(words)):
        #Probability of every tag for particular word
        dummy_prob = []
        for t in T:
            #Transition Pobability
            if i==0:
                tp = TransitionProbs_Trigrams['.!.'][t]
            elif i==1:
                d = predicted_tags[-1]+'!'+'.'
                tp = TransitionProbs_Trigrams[d][t]
            else:
                d = predicted_tags[-1]+'!'+predicted_tags[-2]
                tp = TransitionProbs_Trigrams[d][t]
  
            #Emission Probability
            ep = EmissionProbs[words[i]][t]
            
            #State Probability
            state_prob = tp*ep
            
            dummy_prob.append(state_prob)
            
        max_prob = max(dummy_prob)
        #Getting a tag with max prob
        max_state = T[dummy_prob.index(max_prob)]
        predicted_tags.append(max_state)
    return list(predicted_tags)


# # Hold-out for Trigrams

# In[126]:


print(len(wat))


# In[127]:


## Around 80% of sentences in training and rest in validation


# In[128]:


train_trigrams = wat[0:44116]
validation_trigrams = wat[44116:]


# In[129]:


print(len(train_trigrams))
print(len(validation_trigrams))


# In[130]:


validation_words_trigrams = []
validation_ground_truth_trigrams = []
for sent in validation_trigrams:
    for pair in sent:
        validation_words_trigrams.append(pair[0])
        validation_ground_truth_trigrams.append(pair[1])


# In[134]:


start_time = time.time()
validation_predicted_trigrams = Viterbi_Trigrams(validation_words_trigrams,tagged_words)
end_time = time.time()
difference = end_time - start_time
print('The time taken for predicting tags on validation set Trigrams is : {} secs'.format(difference))


# In[135]:


print('Number of words for validation trigram is           {}'.format(len(validation_words_trigrams)))
print('Number of actual tags for validation trigram is     {}'.format(len(validation_ground_truth_trigrams)))
print('Number of predicted tags for validation trigram is  {}'.format(len(validation_predicted_trigrams)))


# In[172]:


wrong_words_trigrams = compute_wrongs(validation_words_trigrams,validation_ground_truth_trigrams,validation_predicted_trigrams)


# In[173]:


print(len(wrong_words_trigrams))


# In[174]:


print('The Wrongly predited words in Trigrams are:-')
for i in wrong_words_trigrams:
    print(i)


# In[137]:


# for i in range(10):
#     print('{} and {}'.format(validation_ground_truth_trigrams[i],validation_predicted_trigrams[i]))


# In[138]:


start_time = time.time()
confusion_matrix_trigrams = compute_confusion_matrix(validation_ground_truth_trigrams,validation_predicted_trigrams)
end_time = time.time()
difference = end_time - start_time
print('The time taken for computing confusion matrix for Validation Trigrams is : {} secs'.format(difference))


# In[139]:


print(confusion_matrix_trigrams)


# In[142]:


start_time = time.time()
accuracy_trigrams = compute_accuracy(validation_ground_truth_trigrams,validation_predicted_trigrams)
accuracy_trigrams *= 100
end_time = time.time()
diff = end_time - start_time
print('The time taken for computing accuracy for validation trigrams is : {}'.format(diff))
print('THE ACCURACY FOR TRIGRAMS VALIDATION is : {}'.format(accuracy_3))


# In[143]:


tag_details_trigrams,precision_trigrams,recall_trigrams,f1_score_trigrams=compute_vals(confusion_matrix_trigrams)


# In[144]:


print('Precision for Trigram Validation is : {}'.format(precision_trigrams))
print('Recall for Trigram Validation is    : {}'.format(recall_trigrams))
print('F1-score for Trigram Validation is  : {}'.format(f1_score_trigrams))


# In[145]:


for key in tag_details_trigrams:
    print(key,'->',tag_details_trigrams[key])


# In[ ]:





# In[ ]:





# # Code Required for Demo

# In[175]:


def raw_bigrams(words):
    pred_tags = Viterbi(words,tagged_words)
    return pred_tags


# In[176]:


def raw_trigrams(words):
    pred_tags = Viterbi_Trigrams(words,tagged_words)
    return pred_tags


# In[ ]:




