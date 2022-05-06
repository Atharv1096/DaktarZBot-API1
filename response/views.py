from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
#from .models import predictions
#from .serializers import predictionsSerializers
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv(r"response/Training.csv")
# Create your views here.

@api_view(["GET","POST"])
def responsecode(enteredsymptom):
    try:

        def firstout(s):
            drop = [
                'Itching',
                'Skin Rash',
                'Nodal Skin Eruptions',
                'Continuous Sneezing',
                'Shivering',
                'Chills',
                'Joint Pain',
                'Upper Abdomen Pain',
                'Acidity',
                'Ulcers On Tongue',
                'Muscle Wasting',
                'Vomiting',
                'Burning Micturition',
                'Fatigue',
                'Weight Gain',
                'Anxiety',
                'Cold Hands And Feets',
                'Mood Swings',
                'Weight Loss',
                'Restlessness',
                'Lethargy',
                'Sore Throat',
                'Irregular Sugar Level',
                'Cough',
                'High Fever',
                'Sunken Eyes',
                'Breathlessness',
                'Sweating',
                'Dehydration',
                'Indigestion',
                'Headache',
                'Yellowish Skin',
                'Dark Urine',
                'Nausea',
                'Loss Of Appetite',
                'Pain Behind The Eyes',
                'Back Pain',
                'Constipation',
                'Abdominal Pain',
                'Diarrhoea',
                'Mild Fever',
                'Yellow Urine',
                'Yellowing Of Eyes',
                'Acute Liver Failure',
                'Fluid Overload',
                'Enlarged Lymph Nodes',
                'Malaise',
                'Blurred And Distorted Vision',
                'Sputum In Cough',
                'Throat Irritation',
                'Redness Of Eyes',
                'Runny Nose',
                'Congestion',
                'chest Pain',
                'Weakness in Limbs',
                'Fast Heart Rate',
                'Pain In Abdomen',
                'Pain While Passing Stools',
                'Bloody Stool',
                'Itching In Perineal Region',
                'Neck Pain',
                'Dizziness',
                'Cramps',
                'Bruising',
                'Obesity',
                'Swollen Legs',
                'Swollen Blood Vessels',
                'Puffy Face And Eyes',
                'Enlarged Thyroid',
                'Brittle Nails',
                'Swollen Extremeties',
                'Excessive Hunger',
                'Extra Marital Contacts',
                'Dying And Tingling Lips',
                'Slurred Speech',
                'Knee Pain',
                'Hip Joint Pain',
                'Muscle Weakness',
                'Stiff Neck',
                'Swelling Joints',
                'Movement Stiffness',
                'Spining Movements',
                'Loss Of Balance',
                'Unsteadiness',
                'Weakness Of One Body Side',
                'Loss Of Smell',
                'Increase In Frequency Of Urine',
                'Foul Smell Of Urine',
                'Incomplete Emptying Of Urine',
                'Passage Of Gases',
                'Internal Itching',
                'Typhus',
                'Depression',
                'Irritability',
                'Muscle Pain',
                'Altered Sensorium',
                'Red Spots Over Body',
                'Belly Pain',
                'Abnormal Menstruation',
                'Dischromic Patches',
                'Watering From Eyes',
                'Increased Appetite',
                'Polyuria',
                'Family History',
                'Mucoid Sputum',
                'Rusty Sputum',
                'Lack Of Concentration',
                'Visual Disturbance',
                'Receiving Nlood Transfusion',
                'Injecting Drugs Using Unsterile Injection',
                'Coma',
                'Blood In The Vomitus',
                'Distention Of Abdomen',
                'History Of Alcohol Consumption',
                'Fluid Overload',
                'Blood In Sputum',
                'Prominetne Veins On Calf',
                'Palpitations',
                'Painful Walking',
                'Pus Filled Pimples',
                'Blackheads',
                'Scurring',
                'Skin Peeling',
                'Sliver Like Dusting',
                'Small Dents In nails',
                'Inflammotry Nails',
                'Blister',
                'Red Sore Around Nose',
                'Yellow Crust Ooze',
                'White Ring Like Patches',
                'Redness Over The Area',
                'Swelling Over Red Area',
                'Pain Over Swollen Area',
                'Reddish Dots All Over Body',
                ]



            try:
                dropstring = str(drop)
                dropstring = dropstring.replace("[","")
                dropstring = dropstring.replace("]","")
                dropstring = dropstring.replace("'","")
                dropstring = dropstring.replace(",","")

                # split the string
                #wordss = []
                s = s.split(' ')
                out=[]
                for word in s:
                    if word.title() in dropstring:
                        out.append(str(word.title()))
                        index = dropstring.find(word)
                        #print(index)
                #print(out)
                # iterate in words of string
                #for word in s:

                    # if length is even
                        #print(word)
                #        wordss.append(word.title())

                def check(sentence, words):
                    res = [all([k in m for k in words]) for m in sentence]
                    return [sentence[i] for i in range(0, len(res)) if res[i]]

                # Driver code


                
                #print(check(drop, out))
                out2=check(drop, out)
                if not check(drop,out):
                    out2 = []
                    for i in drop:
                        for word in s:
                            if word.title() in i:
                                out2.append(i)
                                print(word)
                if len(out2)>7:
                    return ['out of bound']
                else:
                    return out2
            except:
                print("Internal Error !")
            #print(wordss)
            # Driver Code

            #printWords(s)

        def secondout(input):
            entered_symptom=input
            entered_symptom=entered_symptom.lower()

            if " " in entered_symptom:
                entered_symptom = entered_symptom.replace(" ","_")
            print(entered_symptom)

            dataframe_columns = df.columns
            index=1
            column_list=[]
            final=[]
            for column in dataframe_columns:
                #print("column "+str(index)+" is "+column)
                index=index+1
                if entered_symptom != column:
                    column_list.append(column)
            #print(column_list)
            length_of_column_list=len(column_list)
            #print(length_of_column_list)

            filtered_data_of_first_column = df[df[entered_symptom]==1]
            #print(filtered_data_of_first_column)
            for symptom in column_list:
                filtered_data_of_second_column = df[df[symptom]==1]
            #print(filtered_data_of_second_column)
            #print(filtered_data)


                rows_of_first_column=list(filtered_data_of_first_column.index.values)
                rows_of_second_column=list(filtered_data_of_second_column.index.values)
                if len(rows_of_first_column)<=len(rows_of_second_column):
                    count=len(rows_of_first_column)
                else:
                    count=len(rows_of_second_column)
                #print("minimum number rows of "+entered_symptom +" and "+symptom + " are "+str(count))
                #print(rows_of_first_column)
                #print(rows_of_second_column)
                #print(rows_of_first_column[767])
                for value in range(0,count):
                    if rows_of_first_column[value] == rows_of_second_column[value]:
                        if symptom not in final:
                            if "_" in symptom:
                                symptom=symptom.replace("_"," ")
                            symptom = symptom.title()
                            final.append(symptom)
            return final


        inputsymptoms=json.loads(enteredsymptom.body)
        if "questioninput" in str(inputsymptoms):

            dotrem = str(inputsymptoms['questioninput'])
            exceptionvalues = ['I ','i ','he ','she ','his ','her ','am ','of ','in ','is ','has ','had ','they ','my ']
            if "stomach" in dotrem:
                dotrem=dotrem.replace("stomach","abdomen")
            for val in exceptionvalues:
                if val in dotrem:
                    dotrem=dotrem.replace(val," ")
            if "." in dotrem:
                dotrem=dotrem.replace(".","")

            questionoutput = firstout(dotrem)
           # questionoutput = str(questionoutput)
            #print('"'+str(inputsymptoms['description'])+'"')
        elif "radioinput" in str(inputsymptoms):
            questionoutput = secondout(str(inputsymptoms['radioinput']))

        print(questionoutput)    
        return Response(questionoutput, status=status.HTTP_200_OK)
        
        
        #return JsonResponse(str(questionoutput), safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

        


        
