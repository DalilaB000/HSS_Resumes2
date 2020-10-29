#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geograpy3
import spacy
import os
import spacy
from find_job_titles import FinderAcora
import pandas as pd
import re
import nltk
import fitz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ipykernel import kernelapp as app


# In[2]:



from nltk import word_tokenize, pos_tag, ne_chunk
'''from nltk import RegexpParser
from nltk import Tree
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords'''


# In[813]:


#Load en_core_web_md augmented
nlp = spacy.load('./')
heading_dictionary = pd.DataFrame()
experience_headings = pd.DataFrame()
skills_headings = pd.DataFrame()
education_headings = pd.DataFrame()
heading_dict = {}

# In[889]:


#Get Dictionary of Experience
#This can also be added to the dictionary spacy, and can be updated as needed
def initialize_headings_file():
    global heading_dictionary, experience_headings,skills_headings,education_headings
    global heading_dict
    heading_dictionary = pd.read_csv("../Lower_big_headings_dictionary.csv")
    heading_dictionary.drop_duplicates(inplace = True)
    experience_headings = heading_dictionary[heading_dictionary.Label == "experience"]
    skills_headings = heading_dictionary[heading_dictionary.Label == "skills"]
    education_headings = heading_dictionary[heading_dictionary.Label == "education"]
    heading_dict = heading_dictionary[["Block_Title","label_id"]]
    heading_dict = dict(heading_dict.values.tolist())

# In[5]:



'''
read_pdf_resume: get the path and the name of the pdf resume and read all the pages into a string variable
Parameter:  pdf_document: a full path to a pdf resume
Return: text: all the text in all pages of the pdf document
'''
def read_pdf_resume(pdf_document):
    try:
        doc = fitz.open(pdf_document) 
        text = ''
        for i, page_n in enumerate(doc.pages()):
            page = page_n.getText("text")
            text += " "+page
    except :
        return ('')
    else:
        return(text)


# In[6]:


'''
initial_cleaning:  split documents using return controler \n; remove empty lines
Parameter: doc_2_clean, a string variable containing the whole resume of a given person
'''

#is_symbol: check if the string has only at most 2 characters (so it may be a symbol)
def remove_stopwords(st):
    #stop_words = set(stopwords.words('english'))
    stop_words = set(["of","in","at","and","for","with","der","de"])
    word_tokens = word_tokenize(st)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence

def is_symbol(s):
    res = s.strip()
    l = list(s)
    if (len(l) < 3):
        return(True)
    else:
        return(False)
def remove_symbols(text_l):
    global nlp
    tmp = [x for x in text_l if not is_symbol(x)]
    return(tmp)
def initial_cleaning(doc_2_clean):
    text2 = doc_2_clean.split("\n")
    text2 = [i.strip() for i in text2]
    text2 = list(filter(lambda x: x != '', text2))
    text2 = remove_symbols(text2)
    return (text2)    

def check_for_nns_nnps(st):
    text = st.title()
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for val, pos_v in tag:
        if pos_v in ["NNPS","NNS"]:
            return(True)
    return(False)
# In[7]:


'''
get_date:  Extract dates from resume.  This is specifically for extracting the date in job experience 
( a range of 2 dates).  For education, one can just use Spicy to extract the date 
parameters:
s : a string
'''

def get_date(s):
    #Covers 90% of date patters
    doc = nlp(s)
    date_pattern =  "(\w{3,9}\.*\s+|[0-9]{2}\/)\d{4}\s*(\-|\–|To|to|TO)\s*((P|p|C|c)|(\w{3,9}\.*\s+|[0-9]{2}\/)\d{4})"

    date_pattern2 = "(\d{4}\s*(\-|\–|To|to|TO)\s*(\d{4}|(P|p|C|c)))|(\s+(\d{2}\/\d{4}))"
    data_pattern3 ="\d{4}\-\d{2}\s*(\-|\–|To|to|TO)\s*((P|p|C|c)|\d{4}\-\d{2})"
    tp = re.search(date_pattern,s)
    if not tp:
        tp2 = re.search(date_pattern2,s)
        if not tp2:
            tp3 = re.search(data_pattern3,s)
            if tp3:
                start = tp3.span()[0]
                end  = tp3.span()[1]
                return [start,end, tp3.string[start:end]] 
            else:
                '''for ent in doc.ents:
                    if ent.label_ == "DATE":
                        pos = re.search(ent.text, s)
                        start = pos.start()
                        end = pos.end()
                        return (start, end, ent.text)'''
                return [0,0,""]
        if tp2:
            start = tp2.span()[0]
            end  = tp2.span()[1]
            return [start,end, tp2.string[start:end]]
        
    else:
        start = tp.span()[0]
        end  = tp.span()[1]
        return [start,end, tp.string[start:end]]

'''
get_date_and_remove_it_from_title:  find the date and remove it.
'''    
def get_date_and_remove_it_from_title(st):
    s = st.strip()
    start,end, date_s = get_date(s)
    print(s)
    if date_s:
        if start == 0:
            pat = r'(P|p|C|c)$'             
            match = re.search(pat, date_s)
            if match:
                print("found")
                end = end + 7
                s = s[end:]
            else:
                s = s[len(date_s):].strip()
        else:
            s = s[:start-1]
    else:
        date_s = ""
    return date_s,s





# In[8]:



'''
clean_title_df: initial cleaning of the titles.  Some titles will have tabs instead of space, or some non-digit 
characters
parameters:
s: string
'''
def clean_title_df(s):
    pattern1 = "\|\s|(\:)$"
    pattern2 = "^[a-z]\."
    s = s.replace("\t"," ")
    s = s.replace(u'\xa0', u' ')
    s = re.sub(pattern2,"",s)
    s =  re.sub(pattern1, "", s)
    s = re.sub("^[:xdigit:]","", s)
    s = re.sub('\A[^A-Za-z0-9\s\(\+]+', '', s)
    s = s.strip()
    return(s) 


# In[290]:stop_words = set(stopwords.words('english'))


'''
gather_headings:  get the headings name, and their location for a given resume
Parameters:  doc_list: resume as a list of strings; resume_name: the resume file name
Return: blocks_df: a pandas dataframe with the resume name, position of a heading, and its name
tmp check_if_headings(block)
'''
def check_if_potential_title(st):

    s = st.strip()
    if re.search("\A([a-z]|(A\s+))",s) :
        return(False)
    if re.search("\A(Skills|SKILLS)",s):
        return(True)
    tp = s.split()
    tp= [x for x in tp if x]
    tp = [x.strip() for x in tp]
    if len(tp) >= 2:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(s)
        filtered_sentence = [w for w in word_tokens if w in stop_words]
        left = set(filtered_sentence) - set(["in","of","at","the","for","and","to"])
        if left:
            return(False)
    s_tp = res = re.sub(r'[^\w\s]', '', s)
    tp = s_tp.split()
    all_cap = [x for x in tp if x.isupper()]
    if len(all_cap) == len(tp):
        return(True)

    if re.search("\A[^a-zA-Z1-9+]",s):
        return(False)
    if re.search("\A[A-Z]{1}[a-z]{2,}\.$",s):
        return(False)
    if re.search("([A-Z]{3,}|[A-Z][a-z]{3,})\.$",s):
        return(False)
    s = s.replace("\t", " ")
    s = s.replace("&", "")
    s = s.replace("|", "")
    s = re.sub(r'\([^)]*\)', '', s)
    date_s, s = get_date_and_remove_it_from_title(s)
    if re.search("\s+\d{5}$", s):
        return (False)
    if date_s:
        return (True)
    if re.search("^[0-9]{4}\Z", s):
        return (True)
    if re.search("\A[0-9]{4}",s):
        return(True)
    match = re.findall("[:,]", s)
    pos_q = re.search(":",s)
    pos_c = re.search(",",s)
    if pos_c and pos_q:
        if pos_c.start() > pos_q.start():
            return(False)
    if pos_q:
        tp = re.split("[:\,]",s)
        if len(tp) == 2 or len(tp) == 1:
            return(False)

    if len(match) > 4:
        return(False)

    s = s.strip()
    if re.search("\,$",s):
        return(False)
    if re.search("(\A\(|\)$)",s):
        return(False)
    s = re.sub('[^\w\s]',"",s)
    tp = remove_stopwords(s)
    s = " ".join(tp)
    tp = [x for x in tp if not re.search("[^a-zA-Z]",x)]
    val_p = [x for x in tp if re.search("\A[A-Z]",x)]
    #val_p = [x for x in tp if re.search("[A-Z]+[a-z]+\:*$",x)]
    if not val_p:
        return(False)
    elif len(tp) == len(val_p):
        return(True)
    elif len(val_p)< len(tp):
        return(False)
    else:
        tmp = s.split(" ")
        tmp = [x for x in tmp if not x == '']
        tmp = [x.strip() for x in tmp]
        if (len(tmp) == 1) and tmp[0][0].isupper():
            if re.search("[a-z]+\.$",tmp[0]):
                return(False)
            else:
                return(True)
        
        if (len(tmp) == 2) and (tmp[0][0].isupper()):
             return(True)
        elif (len(tmp) >= 3):
            tmp = [x for x in tmp if x not in ["of","and","for","in","at"]]
            if tmp[0].isupper():
                return(True)
            if tmp[0][0].isupper():
                if tmp[1][0].isupper():
                    return(True)
                else:
                    return(False)
            else:
                return(False)
        else:
            return(False)




def multiple_replace(string, reps, re_flags = 0):
    """ Transforms string, replacing keys from re_str_dict with values.
    reps: dictionary, or list of key-value pairs (to enforce ordering;
          earlier items have higher priority).
          Keys are used as regular expressions.
    re_flags: interpretation of regular expressions, such as re.DOTALL
    """
    if isinstance(reps, dict):
        reps = reps.items()
    pattern = re.compile("|".join("(?P<_%d>%s)" % (i, re_str[0])
                                  for i, re_str in enumerate(reps)),
                         re_flags)
    return pattern.sub(lambda x: reps[int(x.lastgroup[1:])][1], string) 

'''
check_number_uppercases_words:  check if the words in a line are upper case.  However, it makes sure to 
remove digits, some stopwords, that are lower cased in a title.  This function is used after the first pass of 
extracting titles from a resume.  
Parameters:
s: string
Return:  True if a true title, else it returns false'''

def check_number_uppercases_words(s):
    replacements = [("\t", ""), ("&", ""),("\/" , ""),(":",""),(",",""),("-"," "), (" for ", " "),
                    ("–" , ""),(" at "," "),(" of "," "),(" in "," "),(" to "," "),(" and "," "),("[0-9]","")]
    s = multiple_replace(s, replacements)
    tmp = s.split(" ")
    tmp = [x.strip() for x in tmp if x != " "]
    tmp = [x for x in tmp if x]
    upper_val = [x for x in tmp if  x[0].isupper()]
    if (len(tmp) <= 2 & len(upper_val) == 1):
        return(True)
    elif len(tmp) == len(upper_val):
        return(True)
    else:
        return(False) 

'''
gather_headings:  get all the headings (titles) from the resume.
Paramerters:
  doc_list:  reading pdf to text result in a text with return lines, so a split result in a list of strings
  resume_name: the file name of the resume
  
Returns:
  blocks_df: a pandas df
'''   
def gather_headings(doc_list, resume_name):
    blocks_df = pd.DataFrame()
    for i,block in enumerate(doc_list):
        block = clean_title_df(block)
        tmp = block.split()
        if len(tmp) == 1:
            block = block.strip()
            if block[0].isupper():
                tmp_df = pd.DataFrame({"Resume_Name": resume_name,"Block_Pos" : i, "Block_Title" : block},index = [i])
                blocks_df = blocks_df.append(tmp_df)
        elif check_if_potential_title(block):
            tmp_df = pd.DataFrame({"Resume_Name": resume_name,"Block_Pos" : i, "Block_Title" : block},index = [i])
            blocks_df = blocks_df.append(tmp_df)
    blocks_df.reset_index()
    tp = [ x for ix, x in blocks_df.iterrows() if check_number_uppercases_words(x.Block_Title)]
    blocks_df = pd.DataFrame(tp)
    #blocks_df.reset_index()
    return(blocks_df)   


# In[292]:


def V2_gather_headings(doc_list, resume_name):
    blocks_df = pd.DataFrame()
    pattern2ignore = "\A([A-Z][a-z]{3,}\s*){2,3}\."
    for i,block in enumerate(doc_list):
        block = clean_title_df(block)
        tmp = block.split()
        if check_if_potential_title(block):
            tmp_df = pd.DataFrame({"Resume_Name": resume_name,"Block_Pos" : i, "Block_Title" : block},index = [i])
            blocks_df = blocks_df.append(tmp_df) 
    #blocks_df.reset_index()
    #tp = [ x for ix, x in blocks_df.iterrows() if check_number_uppercases_words(x.Block_Title)]
    #blocks_df = pd.DataFrame(tp)
    #blocks_df.reset_index()
    return(blocks_df)   


# In[982]:


# Create an nlp object
from spacy.matcher import Matcher
def maybe_company(s):
    m = s.split()
    if m[0].isupper():
        return(True)
    else:
        return(False)
    
def get_date_or_company(st):
    date_s = ""
    company_s = ""
    s = st.strip()
    doc = nlp(s)
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_s = ent.text
            if not date_s:
                _,end,date_s = get_date(s)  
        elif ent.label_ == "ORG":
            company_s = ent.text
    return(date_s,company_s)
def extract_degree_level(st):
    s = st.strip()

    doc = nlp(s)
    degree = ""
    degree_type = ""
    for ent in doc.ents:
        if ent.label_ == "EDU":
            degree_type = ent.ent_id_
            degree = ent.text
            return degree, degree_type
    return degree, degree_type

def check_if_city_in(s):
    doc = nlp(s)
    for ent in doc.ents:
        if ent.label_ == "GEO":
            return "GEO"
    return ""

def extract_all_info_degree(s):
    degree_dict = {"GPE":"","ORG":"","Bachelor":"","Master":"","Doctor":"","Major":"","DATE":""}
    degree_dict["DATE"], s = get_date_and_remove_it_from_title(s)
    s = re.sub("\s+(c|C)um\s+(l|L)aude","",s)
    s = s.sub("\s+(S|s)uma\s+","",s)
    s = s.sub("(m|M)agna","",s)
    if re.search("\s(A|a)t\s",s):
        tp = re.split("\s+(A|a)t\s+",s)
        l = len(tp)-1
        degree_dict["ORG"] = tp[l]
        s= tp[0]
    doc = nlp(s)
    degree_c = 1
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if not degree_dict["DATE"]:
                degree_dict[ent.label_] = ent.text
                s = s.replace(ent.text,"")
            
        else:
            if ent.label_ == "EDU":
                degree_dict[ent.ent_id_] = ent.text
                s = s.replace(ent.text,"")
            elif ent.label_ == "ORG":
                if not degree_dict["ORG"]:
                        degree_dict["ORG"] = degree_dict["ORG"]+" | "+ent.text
                        s = s.replace(ent.text,"")
            elif ent.label_ == "GPE":
                if not degree_dict["GPE"]:
                        degree_dict["GPE"] = ent.text
                        s = s.replace(ent.text,"")
            
            else:
                if ent.label_ in list(degree_dict.keys()):
                        degree_dict[ent.label_] = ent.text
                        s = s.replace(ent.text,"")
    if not degree_dict["DATE"]:
        tp = re.search("\s\d{4}$",s)
        if tp:
            degree_dict["DATE"]= s[tp.start():]
            s = s[:tp.start()]
           
            
    if not degree_dict["GPE"]:
        degree_dict["GPE"]= extract_city(s)
        if degree_dict["GPE"]:
            l = len(degree_dict["GPE"])
            degree_dict["GPE"] = degree_dict["GPE"][l-1]
            s = s.replace(degree_dict["GPE"],"")
        else:
            degree_dict["GPE"]= ""
    if not degree_dict["Major"]:
        s = s.replace(".","")
        s = re.sub("\,","",s)
        s = s.strip()
        if re.search("School|University|College|Institute",s):
            degree_dict["Major"]= degree_dict["ORG"]
            degree_dict["ORG"] = re.sub("[^a-zA-Z&\s]","",s).strip()
        else:  
            tp = re.split("(IN|in|In)",s)
            if len(tp) > 1:
                degree_dict["Major"]= re.sub("[^[a-zA-Z&\s]","",tp[len(tp)-1]).strip()
            else:
                degree_dict["Major"]= s
        
    return degree_dict





'''def get_degree_university_date(s):
doc = nlp("South Dakota State University, Brookings MA, Economics 2014")
for ent in doc.ents:
    print(ent.label_,ent.text)'''
#city_s = extract_city("Augustana University, Sioux Falls BA, Econ & Math 2012")
#print(major, " * ",city_s," * ",degree, " * ", company_s, " * ",date_s)
#s = "Fordham University, New York City PhD, Economics In progress, ABD"
#extract_city("New York City")
def get_job_info(s):
    found = False
    found_date = False
    job_dict = {"DATE":"","ORG":"","GPE":"","Job_Title":""}
    job_dict["GPE"] = extract_city(s)
    if job_dict["GPE"]:
        l = len(job_dict["GPE"])-1
        job_dict["GPE"] = job_dict["GPE"][l]
        s = s.replace(job_dict["GPE"],"")
    else:
        job_dict["GPE"] = ""
    doc = nlp(s)
    n = 1
    for ent in doc.ents:
        '''if not found:
            start,end, job_title = list(get_job_title(ent.text))[0]
            if job_title:
                job_dict["Job_Title"] = job_title
                found = True'''
        print("s ",s)
        if ent.label_ == "GPE":
            if not job_dict["GPE"]:
                job_dict["GPE"] = ent.text
                s = s.replace(ent.text,"")
        elif ent.label_ == "DATE":
            if n == 1:
                job_dict["DATE"] =  ent.text
                s = s.replace(ent.text,"")
                n += 1
            else:
                job_dict["DATE"] = job_dict["DATE"]+" - "+ent.text
                s = s.replace(ent.text,"")
            found_date = True
        elif ent.label_ == "ORG":
            job_dict[ent.label_] = job_dict[ent.label_]+" : "+ent.text
            s = s.replace(ent.text,"")
            
    if not job_dict["DATE"]:    
        job_dict["DATE"],s = get_date_and_remove_it_from_title(s)
    
    if not found:
        s = s.replace(".","")
        s = re.sub("[A-Z]{2}","",s)
        s = re.sub("[^a-zA-Z\s]","",s)
        job_dict["Job_Title"] = s.strip()
    if isinstance(job_dict["GPE"],list):
        job_dict["GPE"] = ""
    return job_dict


# In[873]:


def extract_city(s):
    places = geograpy3.get_place_context(text = s)
    return (places.cities)


# In[16]:


def count_n_POS(s):
    tmp = s.split(",")
    tp = nltk.pos_tag(nltk.word_tokenize(tmp[0]))
    res =[val for i, val in tp]
    res = set(res) 
    return(res)
def is_phone_number(s):
    res = re.sub(r'[^\w^\s]', '', s) 
    res = res.replace(' ',"")
    l = list(res)
    if res.isnumeric() and (len(l) >= 9):
        return(True)
    else:
        return(False)


# In[899]:


#"from find_job_titles import FinderAcora"

#s = 'SR. DATA PROCESSING SYSTEMS ANALYST'
#Find Job Title
def get_job_title(s):
    f = re.search("(QA\s|ASSISTANT|Assistant|ANALYST|Analyst|PRINCIPAL|Principal)",s)
    if f:
        start, end = f.span(0)
        title_s = f.string
        print(start, end)
        yield start, end, s
    else:
        s = s.title()
        finder=FinderAcora()
        result = finder.finditer(s)
        try:
            gen = iter(result)
            it = next(gen)
            start = it.start
            end   = it.end
            title_s = it.match
            yield start, end, title_s
        except RuntimeError:
            #return start, end, title_s
            yield 0,0,""



# In[193]:


# Teach For America – High School STEM Teacher, Chicago, IL May 2011-December 2012
#Or BOSTON POLICE DEPARTMENT – Boston, MA 2001 TO 2007
'''Pattern if - exist.  In this case, we can have 2 possibilities, either 
the first value is the company or the second is the company'''

'''If date ad city found than we can assume that we have the company ad the job title
'''
def get_city_and_remove_it(st):
    city_s = extract_city(st)
    if city_s:
        city_s = city_s[len(city_s)-1]
        matches = re.finditer(city_s, st)
        matches_positions = [match.start() for match in matches]
        matches_positions = matches_positions[len(matches_positions)-1]
        s = st
        s = s[:matches_positions-1]
        print("Company ",s)
        return city_s, s
    else:
        return "",st


# In[458]:


#In here we assume city and company after job description
#but we have Company, City, KY. - Title

'''
CHECK IT OUT 
'''
def get_title_company_AT(st,pattern_at):
    tp = re.split(pattern_at,st)
    tp = [x.strip() for x in tp if x]
    tp = [x for x in tp if x not in ["at","At"]]
    job_title = tp[0]
    city_s = extract_city(tp[1])
    if city_s and re.search("\,",tp[1]):
        tp_city = re.split("\,",tp[1])
        print(tp_city)
        company_s = tp_city[0]
    return job_title, company_s

'''def get_title_company_commma(st):
    tp = re.split(punct,s)
    tp = [x.strip() for x in tp if x]
    tp = [x for x in tp if x not in [",","|"]]
    start, end, job_title = list(get_job_title(tp[0]))[0]
    if end == 0:
        job_title = tp[1]
        company_s = tp[0]
    else:
        job_title = tp[0]
        company_s = tp[1]
    return job_title, company_s'''

def get_work_heading(st,punct):
    s = st.strip()
    date_s, s = get_date_and_remove_it_from_title(s)
    pattern_at = "\s+(at|At|AT)\s+"
    pattern_comma = "\s*(\,|\||/)\s+"
    pattern_under = "\s*(-|—\|)\s*"
    if re.search(pattern_at,s):
        job_title, company_s = get_title_company_AT(s,pattern_at)
    elif re.search(pattern_under,s):    
        tp = re.split(punct,s)
        tp = [x.strip() for x in tp if x]
        tp = [x for x in tp if x not in ["-","—","|"]]
        print(tp)
        #print(len(tp),"tp " ,tp)
        if len(tp)== 3 :
            tp[1] = tp[len(tp)-1]
        city_s, company_s = get_city_and_remove_it(tp[1])
        if not city_s:
            city_s, company_s = get_city_and_remove_it(tp[0])
            job_title = tp[1]
        if  city_s: 
            job_title = tp[0]
          
                    
    else :
        tp = re.split(pattern_comma,s)
        print(tp)
        tp = [x.strip() for x in tp if x]
        tp = [x for x in tp if x not in [",","|","/"]]
        start, end, job_title = list(get_job_title(tp[0]))[0]
        if end == 0:
            job_title = tp[1]
            company_s = tp[0]
        else:
            job_title = tp[0]
            company_s = tp[1]
    job_dict = {"job Title" : job_title, "Company" : company_s,"Dates" : date_s }
    return(job_dict)
'''#s = "2004-2008 Institute for Educational Sciences Fellow at University of Pennsylvania, Philadelphia, PA"
s = "2011-2013 Director of Research & Evaluation at Camden Coalition of Healthcare Providers, Camden, NJ"
s = "UNIVERSITY OF COLORADO BOULDER at GRADUATE COURSE ASSISTANT"
pattern1 = "(\s+(at|At)\s+|\s*(\–|\-|\—|\|)\s*)"
pattern2 = " at | At "
get_work_heading(s,pattern1)'''

'''RETHINK THIS'''
def get_Title_Co_Commas(st):
    job_title = ""
    company_s = ""
    date_s = ""
    s = st.strip()
    date_s, s = get_date_and_remove_it_from_title(s)
    print(date_s)
    matches = re.finditer("(\s+(a|A)t\s+|\s*(\—|\-)\s*)", s)
    matches_positions = [match.start() for match in matches]
    if matches_positions:
        pattern1 = "(\s+(at|At)\s+|\s*(\–|\-|\—|\|)\s*)"
        res = get_work_heading(s,pattern1)
        job_title = res["job Title"]
        company_s = res["Company"]
    else:
        matches = re.finditer(",|\|", s)
        matches_positions = [match.start() for match in matches]
       
        if matches_positions:
            tp = re.split("(,|\|)",s) 
            tp = [x for x in tp if x not in [",","|"]]
            print(" SSSS ",s, len(tp))
            print(" TPP ",tp)
            if len(tp) == 2:
                _,company_s = get_date_or_company(tp[0])
                if company_s:
                    start,end,job_title = list(get_job_title(tp[1]))[0]
                else:
                    start,end,job_title = list(get_job_title(tp[0]))[0]
                    _, company_s = get_city_and_remove_it(tp[1])
                        
            
                
        else:
            start,end,job_title = list(get_job_title(s))[0]
            if end == 0:
                company_s = s
                
    job_dict = {"job Title" : job_title, "Company" : company_s,"Dates" : date_s }
    return job_dict


# In[197]:


def is_place(s):

    if extract_city(s) == []:
        return (False)
    else:
        return (True)
def is_email(s):
    pat = r'.*?@(.*)\..*'  
    pat = r'[\w.-]+@[\w.-]+'
    match = re.search(pat, s)
    if not match:
        return (False)
    else:
        return(True)
    
def is_github_or_linkedIn(s):
    pattern2 = '(g|G)ithub|(l|L)inked(i|I)n'
    if  any(re.findall(pattern2, s, re.IGNORECASE)):
        return(True)
    else:
        return(False)

def check_for_nns_nnps(st):
    text = st.title()
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for val, pos_v in tag:
        if pos_v in ["NNPS","NNS"]:
            return(True)
    return(False)
# In[199]:
def check_if_heading(st):
    if not isinstance(st, str):
        return("")
    s = st.strip()
    s = s.lower()
    if re.search("\Aeducation$",s):
        return("EDU")
    elif re.search("\A(education\s*(and|\,|&)*)",s):
        return("EDU")
    elif re.search("\Alanguage(s)*(?<!:)$",s):
        return("SKI")
    elif re.search("\A(language(s)*\s*(and|\,|&)*)",s)  \
        or re.search("\A[a-z]*\s+(skills)", s) \
        or re.search("\A(skills$|skills\s+(and|\,|&)*)",s):
        return("SKIL")
    elif re.search("\A(projects$|projects\s+(and|\,|&)*)",s) or \
            re.search("\A[a-z]*\s+(projects)", s):
        return("PRO")
    elif re.search("summary",s):
        return("SUM")
    else:
        if s in heading_dict:
                return(heading_dict[s])
        else:
            return("")
def find_heading_position(resume_df, chosen_heading):
    tmp_l = [x.strip() for x in chosen_heading]
    for i, item in enumerate(resume_df.Block_Title):
        if str(item).lower() in tmp_l:
            return (int(i))
    return(-1)


# In[200]:


def V2_get_start_and_end_experience(resume_df,experience_headings):
    global heading_dictionary
    experience_headings = heading_dictionary[heading_dictionary.Label == "experience"]
    skills_headings = heading_dictionary[heading_dictionary.Label == "skills"]
    education_headings = heading_dictionary[heading_dictionary.Label == "education"]
    resume_df = resume_df.reset_index()
    pos = find_heading_position(resume_df, experience_headings.Block_Title)
    
    if (pos == -1):
        print("Please add the name of the Experience heading to big_heading_dictionary")
        return (0,0)
    else:
        start_t = resume_df.Block_Pos.iloc[pos]+1
        pos = find_heading_position(resume_df, education_headings.Block_Title)
        
        if (pos == -1):
            print("Please add the heading to Education to the big_headings_dictionary.csv")
            education_pos = -1 
        else:
            education_pos = resume_df.Block_Pos.iloc[pos]+1
        pos = find_heading_position(resume_df, skills_headings.Block_Title)
        
        if (pos == - 1):
            skills_pos = -1
        else:
            skills_pos = resume_df.Block_Pos.iloc[pos]+1
                
        if ( start_t < education_pos) & (start_t < skills_pos):
            end_extract_pos = min(education_pos,skills_pos)
        elif ( start_t > education_pos) &(start_t < skills_pos):
            end_extract_pos = skills_pos
        elif ( start_t < education_pos) & (start_t > skills_pos):
            end_extract_pos = education_pos
        else: 
            end_extract_pos = len(resume_l)
        return(start_t,end_extract_pos)


# In[201]:


#Detect start of an experience (Company name, location, position, place)

def V2_detect_start_experience(text_l,pos,n):
    global keep_Experience_headings
    s = text_l[pos]
    s = s.replace("\t",' ')
    i = pos
    not_start = True
    while not_start:
        _,_,date_s = get_date(s)
        if (not is_symbol(s) and (len(count_n_POS(s)) < 4) and i < n) or check_if_potential_title(s):
            if (i >= len(text_l)) :
                not_start = False
            else:
                keep_Experience_headings =keep_Experience_headings.append(pd.DataFrame({"Exp_Heading" : s}, index=[0]), ignore_index = True)
                s = text_l[i]
                s = s.replace("\t",' ')
                
            i += 1
        elif(i >= n):
            not_start = False
        elif (len(count_n_POS(s)) >= 4):
            not_start = False
        
        
    return (i)


# In[342]:


def V3_detect_start_experience(text_l,pos,n):
    s = text_l[pos]
    pos_l = pos
    start_pos =  next(iter(resume_df[resume_df['Block_Pos']== pos_l-1].index), -1)
    while start_pos:
        pos_l += 1
        start_pos = next(iter(resume_df[resume_df['Block_Pos']== pos_l-1].index), -1)
        if pos_l >= n:
            start_pos = False
    return(pos_l-1) 




# In[202]:


#Extract the paragraph or block with the experience 

def extract_experience(text_l,pos,n):
    
    i = pos
    not_end = True
    experience_txt = ""
    #Is lower, and number of words is more than 
    while not_end:
        if i >= n:
            not_end = False
        else:
            s = text_l[i]
            s = s.replace("\t",' ')
            if check_if_potential_title(s) :
                not_end = False
            else: 
                if is_github_or_linkedIn(s) or is_email(s) or is_phone_number(s):
                    i = n
                    not_end = False
                else:
                    experience_txt = experience_txt + "\n"+ s
            i += 1
    return (experience_txt, i-1) 


# In[203]:


def get_all_experiences(resume_l,start_t,end_extract_pos):

    experience_df = pd.DataFrame()
    keep_extracting_exp = True
    len_resume = end_extract_pos
    start_pos = start_t
    counter = 1
    while keep_extracting_exp:
        start_exp = V2_detect_start_experience(resume_l,start_pos,len_resume)
        if start_exp == len_resume - 1:
            keep_extracting_exp = False
        #break
        else:
            exp_txt,end_exp = extract_experience(resume_l,start_exp-1,len_resume)
            start_pos = end_exp
            experience_df = experience_df.append({"Experiences" : exp_txt,"Counter": counter},ignore_index = True)
            counter += 1
            if end_exp >= len_resume - 1:
                keep_extracting_exp = False
            
    return(experience_df)


# In[ ]:


def tag_headings(resume_df):
    for index, row in resume_df.iterrows():
        s = row["Block_Title"]
        degree,degree_type = extract_degree_level(row["Block_Title"])
        start, end,job_title = get_job_title(s)
        pos_degree = re.search(degree,s)
        if pos_degree.start() == 0:
            resume_df[index, "MAJ"] = job_title
            resume_df[index, "Job_Title"] = ""
            doc = nlp(s)
            print(" major ", job_title)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    if n == 1:
                        resume_df.loc[index, "ORG"] = ent.text
                        s = s.replace(ent.text, "").strip()
                        n += 1
                    else:
                        s = s.replace(ent.text, "").strip()
                        resume_df.loc[index, "ORG"] = resume_df.loc[index, "ORG"] + " : " + ent.text
        if degree:
            
            degree_info = extract_all_info_degree(row["Block_Title"])
            #
            #    print(key, " * ",index)
            for key, val in degree_info.items():
                resume_df.loc[index,key] = val
        else:
            job_info = get_job_info(row["Block_Title"])
            print(" JOB INFO ", job_info)
            for key, val in job_info.items():
                resume_df.loc[index,key] = val
            #tp = [1 for x in list(job_info.values()) if x]
            #if tp:
             #   for key, val in job_info.items():
             #       print(key, " * ",index)
             #       resume_df.iloc[index,key] = val
    return resume_df

def parse_all_engineers_resume():
    path_name = "../HSS_Resumes/"
    resume_name = "Engineers Resume_df201026.csv"
    csv_filename = path_name+resume_name
    resume_df = pd.read_csv(csv_filename)
    resume_df.EDU = ""
    resume_df.MAJ = ""
    resume_df.Job_Title = ""
    resume_df.Head_Tag = ""
    resume_df.Subheading = ""
    resume_df = resume_df[1:629]
    for ndx, row in resume_df.iterrows():
        tmp = check_if_heading(row.Block_Title)
        resume_df.loc[ndx, "Head_Tag"] = tmp
    resume_df.to_csv("Analyze_Engineers_500_Resume_df201027.csv")
    print("Finished")
    return ("Complete")
# In[1008]:
nlp = spacy.load('./')
initialize_headings_file()
path_name = "../CFDS/"
resume_name = "125856299.pdf"
pdf_document = path_name+ resume_name
text = read_pdf_resume(pdf_document)
resume_l = initial_cleaning(text)
s = "1999: Ph.D. Mechanical Engineering, The Pennsylvania State University, University Park, PA"
if check_if_potential_title(s):
    print("Title")
resume_df = V2_gather_headings(resume_l,resume_name)
resume_df.to_csv("resume_df_130118433.csv")
print("test")
resume_df.to_csv("resume_df_130118433.csv")
#outcome = parse_all_engineers_resume()

# path_name = "../HSS_Resumes/"
# resume_name = "Engineers Resume_df201026.csv"
#
# keep_Experience_headings = pd.DataFrame()
# pdf_document = path_name+ resume_name
# text = read_pdf_resume(pdf_document)
# resume_l = initial_cleaning(text)
# resume_df = V2_gather_headings(resume_l,resume_name)
# resume_df.EDU = ""
# resume_df.MAJ = ""
# resume_df.Job_Title = ""
# resume_df.Head_Tag = ""
# resume_df.Subheading = ""
#
# for ndx, row in resume_df.iterrows():
#     tmp = check_if_heading(row.Block_Title)
#     if tmp in ["EDU","GEO","DATE"]:
#         resume_df.loc[ndx,"Subheading"] = tmp
#         resume_df.loc[ndx,"Head_Tag"] = ""
#     else:
#         resume_df.loc[ndx,"Head_Tag"] =  tmp
#         resume_df.loc[ndx,"Subheading"]= ""
# resume_df.to_csv("Analyze Resume_df.csv")
#
# start_t, end_extract_pos = V2_get_start_and_end_experience(resume_df,experience_headings)
# resume_df.reset_index(inplace=True,drop=True)
# resume_df2 = tag_headings(resume_df)
# resume_df2.loc[resume_df2.Block_Pos < start_t, "Job_Title"] = ""

#total_experience = get_all_experiences(resume_l,start_t,end_extract_pos)
#resume_df


# In[1011]:


resumes2process = pd.read_csv("../CFDS_Resume_2_Process.csv")


# In[1014]:


os.getcwd()


# In[1017]:


all_job_titles = pd.DataFrame()
path_name = "../CFDS/"
resumes2process = pd.read_csv("../CFDS_Resume_2_Process.csv")
pdf_files = resumes2process.CFDS_Resume
for resume_name in pdf_files:
    keep_Experience_headings = pd.DataFrame()
    pdf_document = path_name+ resume_name 
    text = read_pdf_resume(pdf_document)
    if text:
        resume_l = initial_cleaning(text)
        resume_df = V2_gather_headings(resume_l,resume_name)
        if resume_df.empty:
            print("No Headings")
        else:
            start_t, end_extract_pos = V2_get_start_and_end_experience(resume_df,experience_headings)
            resume_df.reset_index(inplace=True,drop=True)
            resume_df2 = tag_headings(resume_df)
            resume_df2.loc[resume_df2.Block_Pos < start_t, "Job_Title"] = ""
            all_job_titles = all_job_titles.append(resume_df2,ignore_index = True)
    





def add_flags_2_resume_df(resume_df):
    resume_df2 = resume_df.copy()
    
    for index, row in resume_df2.iterrows():
        _,_,job_t = list(get_job_title(row["Block_Title"]))[0]
        date_d, company_s = get_date_or_company(row["Block_Title"])
        _,_,date_s = get_date(row.Block_Title)
        city_s = extract_city(row["Block_Title"])
        degree_level = extract_degree_level(row["Block_Title"])
        if job_t:
            resume_df2.loc[index,"Job_Title"] = 1
            resume_df2.loc[index,"Job_Title_txt"] = job_t
            print("title :", job_t,row["Block_Pos"])
        if company_s:
            resume_df2.loc[index,"Company_Name"] = 1
            resume_df2.loc[index,"Company_Name_txt"] = company_s
            print("Company: ",company_s, row.Block_Pos)
        if date_s:
            resume_df2.loc[index,"Date_s"] = 1
            resume_df2.loc[index,"Date_s_txt"] = date_s
            print("Date: ",date_s,row.Block_Pos)
        if city_s:
            resume_df2.loc[index,"City_Name"] = 1
            resume_df2.loc[index,"City_Name_txt"] = city_s
        if degree_level:
            resume_df2.loc[index,"Degree_level"] = 1
            resume_df2.loc[index,"Degree_level_txt"] = degree_level
        if date_d and not date_s:
            resume_df2.loc[index,"Degree_Date"] = 1
            resume_df2.loc[index,"Degree_Date_txt"] = date_d
    return resume_df2




def tag_headings(resume_df):
    for index, row in resume_df.iterrows():
        degree,degree_type = extract_degree_level(row["Block_Title"])
        if degree:
            
            degree_info = extract_all_info_degree(row["Block_Title"])
            #
            #    print(key, " * ",index)
            for key, val in degree_info.items():
                resume_df.loc[index,key] = val
        else:
            job_info = get_job_info(row["Block_Title"])
            print(" JOB INFO ", job_info)
            for key, val in job_info.items():
                resume_df.loc[index,key] = val
            #tp = [1 for x in list(job_info.values()) if x]
            #if tp:
             #   for key, val in job_info.items():
             #       print(key, " * ",index)
             #       resume_df.iloc[index,key] = val
    return resume_df


# In[1000]:


start_t


# In[419]:


#def get_start_end_continous_Rows(resume_df,)
'''found_block = False

resume_df2 = add_flags_2_resume_df(resume_df.query("index > "+str(start_pos)+" and index < "+str(end_extract_pos)))
resume_df2.reset_index(drop=True, inplace=True)

resume_df2["Sum_Experience_Val"] = resume_df2[["Job_Title","Company_Name","Date_s","City_Name"]].sum(axis = 1)
end_resume_df = False
index = 0
start = 0
len_resume_df2,_ = resume_df2.shape
start_end_df = pd.DataFrame()
for index, row in enumerate(resume_df2):
    if index+1 < len_resume_df2: 
        diff_val = resume_df2.Block_Pos[index+1] - resume_df2.Block_Pos[index]
        if resume_df2.Sum_Val[index] == 0:
            start_end_df = start_end_df.append({"start":resume_df2.Block_Pos[start],"end":resume_df2.Block_Pos[index],
                                               "sum_val":resume_df2.Sum_Experience_Val[start:index+1].sum(axis=0)}, ignore_index = True)
            start = index+1
        else:
            if  diff_val > 2:
                start_end_df = start_end_df.append({"start":resume_df2.Block_Pos[start],"end":resume_df2.Block_Pos[index],
                                                   "sum_val":resume_df2.Sum_Experience_Val[start:index+1].sum(axis=0)},ignore_index = True)
                start = index+1
            
        
        '''
    


# In[420]:


'''start_end_df


# In[421]:


#start_end_df = start_end_df[1:,:]
finished = False
index = 0 
len_df,_ = start_end_df.shape
experience_df = pd.DataFrame()
counter = 1 
for idx in range(0,len_df-1):
    if start_end_df.sum_val[idx] > 0:
        text_data = ""
        for i in range(int(start_end_df.end[idx]+1),int(start_end_df.start[idx+1])):
            text_data = text_data+"\n"+resume_l[i]
        counter += 1
        experience_df = experience_df.append({"Experiences":text_data},ignore_index = True)
        if (counter > 3):
            break
experience_df


# In[374]:


resume_df2 = resume_df2.assign(Sum_Val = 0)
resume_df2["Sum_Val"] = resume_df2[["Job_Title","Company_Name","Date_s","City_Name"]].sum(axis = 1)
'''




# In[ ]:




