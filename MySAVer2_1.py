#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tkinter import *
import tkinter as tk
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pickle

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

filename = 'NBC_model.sav'
loaded_NBC = pickle.load(open(filename, 'rb'))

from bs4 import BeautifulSoup
import requests
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
# from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class Main_window:
    def __init__(self, window):
        
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x500") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To PhuongSA!",
                             font = ('Arial' , 13, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
    
        
        self.text="An Object-Oriented Programming-Based Application"
        self.label = tk.Label(self.window, text = self.text,
                             font=("Helvetica", 8, "bold italic"), fg = 'green')
        self.label.pack()
        self.label = tk.Label(self.window, text = " With The Machine Learning & AI Integration For Sentiment Analysis",
                             font=("Helvetica", 8, "bold italic"), fg = 'green')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0 ",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "(only compatible with Windows 7 or Later) ",
                             font=("Time", 8, "bold italic"), fg = 'red')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'magenta')
        self.label.pack()
        
         # Optional step: insert an image
        self.icon = tk.PhotoImage(file = "phuong.PNG")
        self.label = tk.Label(self.window, image = self.icon)
        self.label.pack()# grid(row=8)
        

        
        
        self.label = tk.Label(self.window, text = "Choose your preferred tool below and",
                             font=("Times", 10, "bold"), fg = 'blue')
        self.label.pack()
        self.label = tk.Label(self.window, text = "Enjoy your AI-based Sentiment Analysis",
                             font=("Times", 10, "bold"), fg = 'blue')
        self.label.pack()
        
        
        # Creating the option button
        
        
    
     
        self.DL_base_button = tk.Button(self.window, text="Named-Entity Recognition",
                                          font=("Times", 9, "bold"), fg = 'green',#bg='green')
                                        command=self.NER_window)
        self.DL_base_button.pack(side = TOP)
        
        
        self.quit_button = tk.Button(self.window, text="Quit",
                                          font=("Times", 9, "bold"), fg = 'red',#,#bg='green')
                                       command=self.ExitApplication)
        self.quit_button.pack(side = BOTTOM)
        
        self.DL_base_button = tk.Button(self.window, text="Aspect-Based Sentiment Analysis",
                                          font=("Times", 9, "bold"), fg = 'green',#bg='green')
                                        command=self.ABSA_window)
        self.DL_base_button.pack(side = BOTTOM, padx=7)
        
        self.DL_base_button = tk.Button(self.window, text="Topic Modelling",
                                          font=("Times", 9, "bold"), fg = 'green',#)
                                        command=self.Topic_Modelling_window)
        self.DL_base_button.pack(side = BOTTOM, padx=7)
        
       
        
        self.rule_base_button = tk.Button(self.window, text="Rule-based System",
                                          font=("Times", 9, "bold"), fg = 'green',
                                       command=self.Rule_based_system_window)
        self.rule_base_button.pack(side = LEFT, padx=7)
        
        self.ML_base_button = tk.Button(self.window, text="Machine Learning System",
                                          font=("Times", 9, "bold"), fg = 'green',
                                       command=self.Machine_Leanrning_system_window)
        self.ML_base_button.pack(side = LEFT, padx=7)
        
        self.DL_base_button = tk.Button(self.window, text="Deep Learning System",
                                          font=("Times", 9, "bold"), fg = 'green',
                                        command=self.Deep_Leanrning_system_window)
        self.DL_base_button.pack(side = LEFT, padx=7)
    
        
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the PhuongSA app?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')
                                
        
        
        
        
        
       
        
        

        
#     def close_windows(self):
#         self.window.destroy()
        
    def Topic_Modelling_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = Topic_Modelling_system(self.newWindow)    

    def Rule_based_system_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = Rule_based_system(self.newWindow)
        
    def Machine_Leanrning_system_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = Machine_Leanrning_system(self.newWindow)
        
    def Deep_Leanrning_system_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = Deep_Leanrning_system(self.newWindow)
        
    def ABSA_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = ABSA_system(self.newWindow)
        
    def NER_window(self):
        self.newWindow = tk.Toplevel(self.window)
        self.app = NER_system(self.newWindow)
        

class Topic_Modelling_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x450") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To Topic Modelling PhuongSA System",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'green')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
     
    # Step 2: Finding the NER based on the Spacy
        
        self.NER_finder = tk.Button(self.window, text="Topic Modelling Analysis",
                                     font=("Times", 9, "bold"), fg = 'green')#,
                                      #command=self.NBC_Based_SA)
                        
        self.NER_finder.pack()
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Results:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.NER_result = tk.Label(self.window, text=" ") 
        self.NER_result.pack()
       
        
        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack(side = BOTTOM) 
        
        
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the Topic Modelling Analysis?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')
                                
        
        

class Rule_based_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x450") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To The Rule-based PhuongSA System",
                             font = ('Arial' , 13, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'green')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
        
        # Step 2: Compute the scores of Polarity and Subjectivity based on the TextBlod
        
        self.btn_convert = tk.Button(self.window, text="Sentiment Analysis",
                                     font=("Times", 9, "bold"), fg = 'green',
                                     command=self.Blod_Based_SA)
                        
        self.btn_convert.pack()
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Polarity:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result1 = tk.Label(self.window, text=" ") 
        self.lbl_result1.pack()
        self.label = tk.Label(self.window, text = "Subjectivity:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result2 = tk.Label(self.window, text=" ") 
        self.lbl_result2.pack()  
        
 


        
        # Optional step: insert an image
        self.icon = tk.PhotoImage(file = "funny.PNG")
        self.label = tk.Label(self.window, image = self.icon)
        self.label.pack()
        

        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack()
    
    def Blod_Based_SA(self): 
        self.text = self.ent_text.get()
        text_blob=TextBlob(self.text)
        self.pol=text_blob.sentiment.polarity
        self.sub=text_blob.sentiment.subjectivity
        self.lbl_result1["text"] =  round(self.pol,4) #round(pol, 2) 
        self.lbl_result2["text"] =  round(self.sub,4)
        if round(self.pol,4) < 0:
            tk.messagebox.showwarning("Warning","Hey! be careful your customer's review is negative")
            

        
    
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the Rule-Based PhuongSA System?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')
        
class Machine_Leanrning_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x450") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To The Machine Learning-based PhuongSA System",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'green')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
        
        
        # Step 2: Compute the scores of Polarity and Subjectivity based on the TextBlod
        
        self.btn_convert = tk.Button(self.window, text="Sentiment Analysis",
                                     font=("Times", 9, "bold"), fg = 'green',
                                      command=self.NBC_Based_SA)
                        
        self.btn_convert.pack()
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Polarity:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result1 = tk.Label(self.window, text=" ") 
        self.lbl_result1.pack()
        self.label = tk.Label(self.window, text = "Probability:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result2 = tk.Label(self.window, text=" ") 
        self.lbl_result2.pack()  
        
        # Optional step: insert an image
        self.icon = tk.PhotoImage(file = "funny.PNG")
        self.label = tk.Label(self.window, image = self.icon)
        self.label.pack()
        

        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack() 
        
    def NBC_Based_SA(self):
        self.text = self.ent_text.get()
        self.pol=loaded_NBC.prob_classify(self.text)
        self.lbl_result1["text"]=  self.pol.max()
        self.lbl_result2["text"]=round(self.pol.prob("pos"), 2)
        
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the Machine Learning System?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')

class Deep_Leanrning_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x450") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To The Deep Learning-based PhuongSA System",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'green')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
        
        
        # Step 2: Compute the scores of Polarity and Subjectivity based on the TextBlod
        
        self.btn_convert = tk.Button(self.window, text="Sentiment Analysis",
                                     font=("Times", 9, "bold"), fg = 'green',
                                      command=self.NBC_Based_SA)
                        
        self.btn_convert.pack()
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Polarity:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result1 = tk.Label(self.window, text=" ") 
        self.lbl_result1.pack()
        self.label = tk.Label(self.window, text = "Probability:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.lbl_result2 = tk.Label(self.window, text=" ") 
        self.lbl_result2.pack()  
        
        # Optional step: insert an image
        self.icon = tk.PhotoImage(file = "funny.PNG")
        self.label = tk.Label(self.window, image = self.icon)
        self.label.pack()
        

        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack() 
        
    def NBC_Based_SA(self):
        self.text = self.ent_text.get()
        self.pol=loaded_NBC.prob_classify(self.text)
        self.lbl_result1["text"]=  self.pol.max()
        self.lbl_result2["text"]=round(self.pol.prob("pos"), 2)
        
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the Deep Learning System?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')


class ABSA_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x450") #(width,heigth)
        self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To Apsect-Based Sentiment Analysis (ABSA)",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "PhuongSA System",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'green')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
     
    # Step 2: Finding the NER based on the Spacy
        
        self.NER_finder = tk.Button(self.window, text="Aspect-Based SA",
                                     font=("Times", 9, "bold"), fg = 'green')#,
                                      #command=self.NBC_Based_SA)
                        
        self.NER_finder.pack()
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Results:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.NER_result = tk.Label(self.window, text=" ") 
        self.NER_result.pack()
       
        
        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack(side = BOTTOM) 
        
        
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the ABSA Analysis?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')
                        
            
class NER_system:
    def __init__(self, window,*args):
        self.window = window
        self.window.title("Phuong's Sentiment Analysis Dashboard")
        self.window.geometry("430x510") #(width,heigth)
        #self.window.resizable(0, 0)
        
        self.label = tk.Label(self.window, text = "Welcome To Named-Entity Recognition (NER) PhuongSA System",
                             font = ('Arial' , 10, 'bold'), fg = 'blue',height = 1)#, width = 11,  
                              #borderwidth = 1, relief = 'solid')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Version 2020.2.0",
                             font=("Time", 8, "bold italic"), fg = 'blue')
        self.label.pack()
        
        self.label = tk.Label(self.window, text = "Copyright @ Phuong Nguyen",
                             font=("Helvetica", 9, "bold"), fg = 'red')
        self.label.pack()
        
        
        
        # Step 1: Creating the Entry Widget
        ### Enter the text manually
        self.label = tk.Label(self.window, text = "Enter your text",
                              font=("Times", 9, "bold"), fg = 'magenta')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.ent_text = tk.Entry(self.frm_entry, width=60) 
        self.ent_text.grid(row=0)
        
        self.Manual_NER_finder = tk.Button(self.window, text="Looking for NER based on the entered text",
                                     font=("Times", 9, "bold"), fg = 'green',
                                      command=self.EnterText_NER_SA)
                        
        self.Manual_NER_finder.pack() #side = LEFT, padx=7
        
        
        ### ENter the URL
        self.label = tk.Label(self.window, text = "Or Enter your URL",
                              font=("Times", 9, "bold"), fg = 'magenta')
        self.label.pack()
        self.frm_entry = tk.Frame(self.window,width=50, height=50) 
        self.frm_entry.pack()
        self.url_text = tk.Entry(self.frm_entry, width=60) 
        self.url_text.grid(row=0)
        
        
        
        
     
    # Step 2: Finding the NER based on the Spacy
        
        self.URL_NER_finder = tk.Button(self.window, text="Looking for NER based on URL",
                                     font=("Times", 9, "bold"), fg = 'green',
                                      command=self.URL_NER_SA)
                        
        self.URL_NER_finder.pack() #side = LEFT, padx=7
        
        # Step 3: show the result
        self.label = tk.Label(window, text = "Name of Entities:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()

       
        self.NER_result1 = tk.Label(self.window, text=" ") 
        self.NER_result1.pack()
        self.label = tk.Label(self.window, text = "Types and Numers of Entities:",font=("Times", 9, "bold"),fg = 'blue')
        self.label.pack()
        self.NER_result2 = tk.Label(self.window, text=" ") 
        self.NER_result2.pack()  
        
        # Optional step: insert an image
        self.icon = tk.PhotoImage(file = "funny.PNG")
        self.label = tk.Label(self.window, image = self.icon)
        self.label.pack()
        

        # The quit button
        self.quitButton = tk.Button(self.window, text = 'Quit',font=("Times", 9, "bold"),
                                    fg='red', width = 25, 
                                    command = self.ExitApplication)
        self.quitButton.pack() 
    
    def EnterText_NER_SA(self):
        self.text = self.ent_text.get()
        self.article=nlp(self.text)
        self.NER_result1["text"]=  self.article.ents
        self.labels = [x.label_ for x in self.article.ents]
        self.NER_result2["text"]=Counter(self.labels)
        
    def URL_NER_SA(self):
#         self.text = self.ent_text.get()
        self.url =self.url_text.get()
        self.res = requests.get(self.url)
        self.html = self.res.text
        self.soup = BeautifulSoup(self.html, 'html5lib')
        for script in self.soup(["script", "style", 'aside']):
            script.extract()
        self.text=" ".join(re.split(r'[\n\t]+', self.soup.get_text()))
        
        #self.text = self.url_to_string(self.url)
        self.article=nlp(self.text)
        self.NER_result1["text"]=  self.article.ents
        self.labels = [x.label_ for x in self.article.ents]
        self.NER_result2["text"]=Counter(self.labels)
             
    def ExitApplication(self):
        MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the NER Analysis?',
                                            icon = 'warning')
        if MsgBox == 'yes':
             self.window.destroy()
        else:
            tk.messagebox.showinfo('Return','You will now return to the application screen')
            

def main(): 
    root = tk.Tk()
    app = Main_window(root)
    root.mainloop()

if __name__ == '__main__':
    main()

