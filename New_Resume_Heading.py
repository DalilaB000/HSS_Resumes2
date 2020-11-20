#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geograpy3
import os
import spacy
from find_job_titles import FinderAcora
import pandas as pd
import re
import nltk
import fitz
from nltk.corpus import stopwords
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from nltk.tokenize import word_tokenize
from ipykernel import kernelapp as app
import pytesseract as pt
import pdf2image


from nltk import word_tokenize, pos_tag, ne_chunk





# Load en_core_web_md augmented
nlp = spacy.load('./')
heading_dictionary = pd.DataFrame()
experience_headings = pd.DataFrame()
skills_headings = pd.DataFrame()
education_headings = pd.DataFrame()
heading_dict = {}
edu_list = pd.DataFrame()
job_list = pd.DataFrame()
master_job_df = pd.DataFrame()
master_ski_df = pd.DataFrame()
master_edu_df = pd.DataFrame()
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/\
fine-grained-ner.2020-06-24.tar.gz")
skill_list = ""




# Get Dictionary of Experience
# This can also be added to the dictionary spacy, and can be updated as needed
'''
initialize_headings_file()
Initialize the heading dictionary
'''
def initialize_headings_file():
    global heading_dictionary, experience_headings, skills_headings, education_headings
    global heading_dict
    heading_dictionary = pd.read_csv("../Lower_big_headings_dictionary.csv")
    heading_dictionary.drop_duplicates(inplace=True)
    experience_headings = heading_dictionary[heading_dictionary.Label == "experience"]
    skills_headings = heading_dictionary[heading_dictionary.Label == "skills"]
    education_headings = heading_dictionary[heading_dictionary.Label == "education"]
    heading_dict = heading_dictionary[["Block_Title", "label_id"]]
    heading_dict = dict(heading_dict.values.tolist())





'''
read_pdf_resume: get the path and the name of the pdf resume and read all the pages into a string variable
Parameter:  pdf_document: a full path to a pdf resume
Return: text: all the text in all pages of the pdf document
This was the initial function to reading directly pdf files into a text file; unfortunately, 
all the pdf converters to text libraries has issue with pdf with special fonts, ot type
'''


def read_pdf_resume_old(pdf_document):
    try:
        doc = fitz.open(pdf_document)
        text = ''
        for i, page_n in enumerate(doc.pages()):
            page = page_n.getText("text")
            text += " " + page
    except:
        return ('')
    else:
        return (text)




def read_pdf_resume(pdf_document):
    '''
    read_pdf_resume:  take a pdf file transform it into an image and then ocr to a text
    :param pdf_document: the pdf file name with its path
    :return: text (string)

    '''
    pages = pdf2image.convert_from_path(pdf_path=pdf_document, dpi=300, size=(1654, 2340))
    content = ""
    for page in pages:
        content = content+"\n"+pt.image_to_string(page)
    return content



def remove_stopwords(st):
    '''
    remove_stopwords: get a string and remove stopwords.  Very useful for detecting titles, as
    stopwords are kept lowercase in English
    :param st: a string
    :return: string with no stopwords
    '''
    stop_words = set(["of", "in", "at", "and", "for", "with", "der", "de"])
    word_tokens = word_tokenize(st)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def is_symbol(s):
    '''
    check if the string has less than 3 characters.  If it does, return true.  This is done, as when
    I used the pdf to text libraries, I was getting junk characters
    :param s: string
    :return: boolean
    '''
    res = s.strip()
    l = list(s)
    if (len(l) < 3):
        return (True)
    else:
        return (False)


def remove_symbols(text_l):
    '''
    remove from a list, symbols
    :param text_l: a list
    :return: cleaned list
    '''
    global nlp
    tmp = [x for x in text_l if not is_symbol(x)]
    return (tmp)


def initial_cleaning(doc_2_clean):
    '''
    initial_cleaning:  read the text, split it by return line, and remove
    all empty strings, and symbols.
    Note:  while I could have kept the blank lines as a reference; unfortunately, such format is
    not followed by all resumes
    :param doc_2_clean: string
    :return: a list of strings (lines in a resume)
    '''
    text2 = doc_2_clean.split("\n")
    text2 = [i.strip() for i in text2]
    text2 = list(filter(lambda x: x != '', text2))
    text2 = [re.sub("\s{2,}", " ", x) for x in text2]
    text2 = remove_symbols(text2)
    return (text2)


def rreplace(s, old, new, occurrence):
    '''
    rreplace: replace starting from the end of string
    :param s: the original string
    :param old: old value
    :param new: new value
    :param occurrence: number of changes
    :return: string
    '''
    li = s.rsplit(old, occurrence)
    return new.join(li)


def check_for_nns_nnps(st):
    '''
    check if tags are only noun
    :param st: string
    :return: boolean
    '''
    text = st.title()
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for val, pos_v in tag:
        if pos_v in ["NNPS", "NNS"]:
            return (True)
    return (False)




def get_date(st):
    '''
    get_date: extract the date from a string. It covers over 95% of potential dates in a resume
    month can be:
        * a digit 1-31 or 2 digits 01-02...-31
        * abbreviated month name, or full month name
    year can be:
        * four digits or 'dd
    it can also extract a range of date:
    a range of dates are taken a one long string so if date is as 10/2002 - 12/2020 in a string
    the whole range is extracted
    a range can be defined by to or -, but there are many type of -
    for all other dates, like summer 2019, not captured by get_date, we have another function to extract them
    :param st: a string
    :return: a date string
    '''
    # Covers 90% of date patters
    s = st.title()
    months = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)([a-z]{0,6})?\.?\s*"
    tp = re.search("("+months+"|\d{1,2})?(\/|\.)?\s*(1|2)\d{3}\s?(-|–|—|—|to|TO|To)?\s?(P|p|C|c|("+months+"|\d{1,2})?(\/|\.)?\s*(1|2)\d{3})?",s)
    if not tp:
        tp2 = re.search("(\d{4}\s*(\-|\–|—|To|to|TO)\s*((1|2)\d{3}|(P|p|C|c)))|(\s+(\d{2}\/(1|2)\d{3}))", s)
        if not tp2:
            tp3 = re.search("(1|2)\d{3}\-\d{2}\s*(\-|\–|—|To|to|TO)\s*((P|p|C|c)|(1|2)\d{3}\-\d{2})", s)
            if tp3:
                start = tp3.span()[0]
                end = tp3.span()[1]
                return [start, end, tp3.string[start:end]]
            else:
                tp4 = re.search("^((1|2)\d{3}\s?(\/\d{1,2})?|\d{2}\s?(\/(1|2)\d{3}))\s*",s)
                if tp4:
                    start = tp4.span()[0]
                    end = tp4.span()[1]
                    return [start, end, tp4.string[start:end]]
                else:
                    tp5= re.search("[a-zA-Z]{3,9}\.?('|`|’)\d{2}\s?(\-|\–|—|To|to|TO)\s+([a-zA-Z]{3,9}\.?('|`|’)\d{2}|\
                    (P|p|C|c))",s)
                    if tp5:
                        start = tp5.span()[0]
                        end = tp5.span()[1]
                        return [start, end, tp5.string[start:end]]
                return [0, 0, ""]
        else:
            start = tp2.start()
            end = tp2.end()
            return [start, end, tp2.string[start:end]]

    else:
        start = tp.span()[0]
        end = tp.span()[1]
        return [start, end, tp.string[start:end]]

def get_candidate_name(resume_l):
    '''
    get_candidate_name: extract the candidate name from resume_l, a list of string lines in
    a resume
    :param resume_l: a list of string lines in a resume
    :return: candidate name
    '''
    st = resume_l[0]
    doc = nlp(st)
    p_name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            p_name = ent.text
    return p_name
'''
get_date_and_remove_it_from_title:  find the date and remove it.
'''


def get_date_and_remove_it_from_title(st):
    '''
    get_date_and_remove_it_from_title: in some instances, like getting job info, and education,
    one may want to remove the date
    :param st: string
    :return: date, and string with no date
    '''
    s = st.strip()
    start, end, date_s = get_date(s)
    if date_s:
        if start == 0:
            pat = r'(P|p|C|c)$'
            match = re.search(pat, date_s)
            if match:
                print("found")
                end = end + 7
                s = s[end:]
            else:
                s = s[end:].strip()
        else:
            pat = r'(P|p|C|c)$'
            match = re.search(pat, date_s)
            if match:
                print("found")
                end = end + 7
            s = s[:start] + s[end:]
    else:
        date_s,s = get_date_ent(s)
        if not re.search("\d{4}",s):
            date_s = ""
    return date_s, s


def get_date_ent(st):
    '''
    get_date_ent: use spacy to extract dates that cannot be extracted by get_date like summer 2019.
    Unfortunately, neither spacy nor allennlp are able to extract dates properly (which explains the
    reason for needing get_date)
    :param st: string
    :return: date_s (date) and leftover string
    '''
    doc = nlp(st)
    s = st
    date_s = ""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_s = ent.text
            s = st.replace(date_s, "")
            return date_s, s
    return (date_s, s)



def get_city_allennlp_and_remove_it(st):
    '''
    get_city_allennlp_and_remove_it: get a string and extract the city if it is in it
    :param: st: string
    :return city, and remaining of the string
    '''
    s = st
    city_s = ""
    tp = predictor.predict(
        sentence= st
    )
    for word, tag in zip(tp["words"], tp["tags"]):
        if tag[2:] == "GPE":
            city_s = word
            s = rreplace(s,word,"",1)
    states = '\s*(AZ|BC|AL|AK|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|LA|ME|MD|MA|MI|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)([^a-zA-Z0-9]|\Z)'
    s = re.sub(states,"",s).strip()
    s = re.sub(",$","",s).strip()
    s = s.strip()
    s= re.sub("[^a-zA-Z0-9]$","",s).strip()
    return(city_s, s)

def get_company_allennlp(st):
    '''
    get_company_allennlp: extract company name from a string and return the string
    without the company name
    :param st: string
    :return: company name, and remaining string
    '''
    s = st
    tp = predictor.predict(
        sentence=st
    )
    org_s = ""
    for word, tag in zip(tp["words"], tp["tags"]):
        if tag[2:] == "ORG":
            if not org_s:
                org_s = word
                s = s.replace(word, "")
            else:
                org_s = org_s + " "+word
                s = s.replace(word,"")
    return org_s,s

def get_skills_info(resume_df,resume_l,skill_start):
    '''
    get_skills_info: given the title dataframe, and the resume_l, and the starting position
    of the skill section extract the skills
    :param resume_df: dataframe with titles, sections and their position
    :param resume_l: list of resume lines
    :param skill_start: position in resume_df of skills section
    :return: dataframe with skills, and position in resume_df
    '''
    global skill_list
    in_skill = True
    i = skill_start+1
    len_res = resume_df.shape[0]
    not_end_line = True
    while in_skill and not_end_line:
        if i == len_res:
            not_end_line = False
        elif check_if_heading(resume_df.Block_Title[i]):
            in_skill = False
        else:
            i += 1
    start_extract = resume_df.Block_Pos[skill_start]
    if not_end_line:
        end_extract = resume_df.Block_Pos[i-1]
        tmp_res = resume_l[start_extract:end_extract]
    else:
        tmp_res = resume_l[start_extract:]
    skill_list =""
    for i,skill_val in enumerate(tmp_res):
        s = re.sub("[A-Za-z]'s","s",skill_val)
        pos = re.search(":\s+",s)
        if pos:
            s = s[pos.end():]
        s = re.sub("[^a-zA-Z0-9\s,\.\$%\&\"]",":",s)
        if not skill_list:
            skill_list = s
        else:
            skill_list = skill_list +";"+s
    skill_df = pd.DataFrame([{"SKI": skill_list,"Resume_Name":resume_df.Resume_Name[skill_start]}])
    return skill_df,i

def get_job_info(resume_df,resume_l,work_start):
    '''
    get_job_info: extract company name, job title, date, location, and experience (text paragraph)
    :param resume_df: dataframe with titles and their location in resume_l
    :param resume_l: list of lines (strings) in resume
    :param work_start: starting position of job desc in resume_df
    :return: dataframe, and position reached in resume_df
    '''
    job_list = pd.DataFrame()
    job_dict = {"ORG":"","JOB":"","DATE":"","GPE":"","EXP":""}
    #work_start = 3
    in_job = True
    i = work_start
    last_pos =0
    assume_job = False
    while in_job:
        company_s = ""
        city_s = ""
        job_title = ""
        s = resume_df.Block_Title[i]
        s = re.sub("('|’)(s|i)", "s", s)
        s = re.sub("\s{1,}", " ", s)
        s = re.sub("\s+[b-hj-z]\s+",", ",s)
        s = re.sub("\s+gg\s+",",",s)
        if not re.search("[a-z]\.com",s):
            s = re.sub("\.","",s)
        date_s, s = get_date_and_remove_it_from_title(s)

        s = re.sub(r'\([^)]*\)', '', s).strip()
        s = re.sub(r'\([^)]*', '', s).strip()

        if len(s) <= 2:
            s = ""
        if s:
            city_s, s = get_city_allennlp_and_remove_it(s)



            #s = re.sub("\s*(\||-|-|—|-|\:|\=|\@)\s*",", ",s).strip()
            s = re.sub("\s*[^a-zA-Z0-9\,]\s+", ", ", s).strip()
            s = re.sub("\s+"," ",s).strip()
            s = re.sub("(\A,|,$)","",s).strip()
            s = re.sub("(,\s?){2,}",", ",s)
            start, end, job_title = list(get_job_title(s))[0]

            if job_title:
                job_title = s[start:end]
                if not re.search("\s+(At|AT|at)\s+",s) and not re.search(",",s):
                    if assume_job and job_dict["JOB"]:
                        job_dict["JOB"]= s
                        assume_job = False
                    elif job_dict["JOB"]:
                        if last_pos == 0:
                            last_pos = resume_df.Block_Pos[i]+1
                        else:
                            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                            last_pos = resume_df.Block_Pos[i]+1
                        job_list = job_list.append(job_dict, ignore_index=True)
                        for key in job_dict:
                            job_dict[key]=""

                    job_title = s
                    job_dict["JOB"] = s
                elif re.search("\s+(At|AT|at)\s+",s) and not re.search(",",s):
                    tp = re.split("\s+(At|AT|at)\s+",s)
                    tp = [x.strip() for x in tp if x]
                    tp = [x for x in tp if x not in ["at","AT","At"]]
                    if not job_dict["JOB"] and not job_dict["ORG"]:
                        job_title = tp[0]
                        job_dict["JOB"] = job_title
                        job_dict["ORG"] = tp[1]
                        last_pos = resume_df.Block_Pos[i]+1
                    elif job_dict["JOB"] and not job_dict["ORG"]:
                        job_dict["ORG"] = s
                        last_pos = resume_df.Block_Pos[i]+1
                    elif job_dict["JOB"] and  job_dict["ORG"]:
                        if last_pos == 0:
                            last_pos = resume_df.Block_Pos[i]+1
                        else:
                            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                            last_pos = resume_df.Block_Pos[i]+1
                        job_list = job_list.append(job_dict, ignore_index=True)
                        for key in job_dict:
                            job_dict[key] = ""

                        job_title = tp[0]
                        job_dict["JOB"] = job_title
                        job_dict["ORG"] = tp[1]
                elif re.search(",",s):
                    tp = s.split(",")
                    tp = [x.strip() for x in tp if x]
                    job_t = [x for x in tp if re.search(job_title,x)]
                    if job_t:
                        job_title = job_t[0]
                    s = s.replace(job_title,"")
                    if not job_dict["JOB"] and job_dict["ORG"]:
                            job_dict["JOB"] = job_title
                            last_pos = resume_df.Block_Pos[i] + 1
                    elif not job_dict["JOB"] and not job_dict["ORG"]:
                        job_dict["JOB"] = job_title
                        doc = nlp(s)
                        company_s = ""
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                company_s = ent.text
                        tp.remove(job_title)
                        job_dict["ORG"] = company_s
                        last_pos = resume_df.Block_Pos[i]+1

                    elif job_dict["JOB"] and job_dict["ORG"]:
                        if last_pos == 0:
                            last_pos = resume_df.Block_Pos[i]+1
                        else:
                            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                            last_pos = resume_df.Block_Pos[i]+1
                        job_list = job_list.append(job_dict, ignore_index=True)
                        for key in job_dict:
                            job_dict[key] = ""
                        job_dict["JOB"] = job_title
                        doc = nlp(s)
                        company_s = ""
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                company_s = ent.text
                        tp = [ x for x in tp if x != job_title]
                        #tp.remove(job_title)
                        if company_s:
                            job_dict["ORG"] = company_s

                    elif job_dict["JOB"]:
                        company_s = job_dict["ORG"]
                        if last_pos == 0:
                            last_pos = resume_df.Block_Pos[i]+1
                        else:
                            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                            last_pos = resume_df.Block_Pos[i]+1
                        job_list = job_list.append(job_dict, ignore_index=True)
                        for key in job_dict:
                            job_dict[key] = ""

                        job_dict["JOB"] = job_title
                        doc = nlp(s)
                        company_s = ""
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                company_s = ent.text
                        job_dict["ORG"] = company_s




            else:
                if re.search("\s+(At|AT|at)\s+",s) and not re.search(",",s):
                    tp = re.split("\s+(At|AT|at)\s+",s)
                    tp = [x.strip() for x in tp if x]
                    tp = [x for x in tp if x not in ["at","AT","At"]]
                    company_s = tp[1]
                    if job_dict["ORG"]:
                        if last_pos == 0:
                            last_pos = resume_df.Block_Pos[i]+1
                        else:
                            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                            last_pos = resume_df.Block_Pos[i]+1
                        job_list = job_list.append(job_dict, ignore_index=True)
                        for key in job_dict:
                            job_dict[key] = ""

                    job_dict["ORG"] = company_s
                    if job_dict["ORG"] and not job_dict["JOB"]:
                        job_dict["JOB"] = tp[0]
                        last_pos = resume_df.Block_Pos[i]+1

                elif not re.search("\s+(At|AT|at)\s+",s) and not re.search(",",s):
                    if city_s:
                        company_s = s
                        if job_dict["ORG"]:
                            if last_pos == 0:
                                last_pos = resume_df.Block_Pos[i] + 1
                            else:
                                job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos,
                                                                              resume_df.Block_Pos[i])
                                last_pos = resume_df.Block_Pos[i] + 1
                            job_list = job_list.append(job_dict, ignore_index=True)
                            for key in job_dict:
                                job_dict[key] = ""

                        job_dict["ORG"] = company_s
                    else:
                        doc = nlp(s)
                        company_s = ""
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                if not company_s:
                                    company_s = ent.text
                                else:
                                    company_s = company_s + " " + ent.text
                        if company_s:
                            if job_dict["ORG"]:
                                if last_pos == 0:
                                    last_pos = resume_df.Block_Pos[i] + 1
                                else:
                                    job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos,
                                                                                  resume_df.Block_Pos[i])
                                    last_pos = resume_df.Block_Pos[i] + 1
                                job_list = job_list.append(job_dict, ignore_index=True)

                                for key in job_dict:
                                    job_dict[key] = ""

                            job_dict["ORG"] = s
                        else:
                            if job_dict["ORG"] and not job_dict["JOB"]:
                                job_dict["JOB"] = s
                                assume_job = True
                                last_pos = resume_df.Block_Pos[i]+1
                            elif not job_dict["ORG"] and  job_dict["JOB"]:
                                job_dict["ORG"] = s
                                last_pos = resume_df.Block_Pos[i]+1
                            else:
                                unknown =s

                elif re.search(",",s):
                    s = re.sub("(\A,|,$)"," ",s).strip()
                    tp = s.split(",")
                    company_s, s = get_company_allennlp(s)
                    if not company_s:
                        doc = nlp(s)
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                if not company_s:
                                    company_s = ent.text
                                    s = s.replace(ent.text,"")
                                else:
                                    company_s = company_s+"; "+ company_s
                                    s = s.replace(ent.text,"")
                    s = s.strip()

                    s = re.sub("[^a-zA-Z0-9]"," ",s).strip()
                    if len(s) <= 3:
                        s = ""
                    # doc = nlp(s)
                    # company_s = ""
                    # company_tmp = []
                    # p = 0
                    # for ent in doc.ents:
                    #     if ent.label_ == "ORG":
                    #         company_tmp.append(ent.text)
                    #         s = s.replace(ent.text,"").strip()
                    if company_s and s:
                        if not job_dict["ORG"] and not job_dict["JOB"]:

                            job_dict["JOB"] = s
                            assume_job = True
                            job_dict["ORG"] = company_s
                            last_pos = resume_df.Block_Pos[i]+1
                        elif job_dict["ORG"] and job_dict["JOB"]:
                            if last_pos == 0:
                                last_pos = resume_df.Block_Pos[i] + 1
                            else:
                                job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos,
                                                                              resume_df.Block_Pos[i])
                                last_pos = resume_df.Block_Pos[i] + 1
                            job_list = job_list.append(job_dict, ignore_index=True)
                            for key in job_dict:
                                job_dict[key] = ""
                            job_dict["JOB"] = s
                            assume_job = True
                            job_dict["ORG"] = company_s
                    elif company_s and not s:
                        if not job_dict["ORG"]:
                            job_dict["ORG"] = company_s
                        else:
                            if last_pos == 0:
                                last_pos = resume_df.Block_Pos[i] + 1
                            else:
                                job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos,
                                                                              resume_df.Block_Pos[i])
                                last_pos = resume_df.Block_Pos[i] + 1
                            job_list = job_list.append(job_dict, ignore_index=True)
                            for key in job_dict:
                                job_dict[key] = ""
                            job_dict["ORG"] = company_s

                        if not job_dict["JOB"] and s and date_s:
                            job_dict["JOB"] = s
                            assume_job = True
                            last_pos = resume_df.Block_Pos[i]+1


        if date_s:
            if job_dict["DATE"]:
                if last_pos == 0:
                    last_pos = resume_df.Block_Pos[i] + 1
                else:
                    job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos, resume_df.Block_Pos[i])
                    last_pos = resume_df.Block_Pos[i] + 1
                job_list = job_list.append(job_dict, ignore_index=True)
                for key in job_dict:
                    job_dict[key]=""
            last_pos = resume_df.Block_Pos[i] + 1
            job_dict["DATE"] = date_s
        i += 1
        if i == resume_df.shape[0]:
            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos + 1, resume_df.shape[0]-1)
            job_list = job_list.append(job_dict, ignore_index=True)
            job_list["Resume_Name"] = resume_df.Resume_Name[work_start]
            in_job = False
        elif check_if_heading(resume_df.Block_Title[i]):
            print(last_pos)
            print(resume_df.Block_Pos[i])
            job_dict["EXP"] = extract_job_experience_text(resume_l, last_pos+1, resume_df.Block_Pos[i])
            job_list = job_list.append(job_dict, ignore_index=True)
            job_list["Resume_Name"]= resume_df.Resume_Name[i]
            in_job = False

    return job_list

def get_education_info(resume_df, edu_start):
    '''
    get_education_info: get education info from resume_df
    :param resume_df: dataframe with titles
    :param edu_start: start of education section
    :return: education dataframe with all the educations, and position in resume_df
    '''
    global edu_list
    degree_dict = {"ORG": "", "DEGREE": "", "MAJ": "", "DATE": "", "GPA": ""}
    edu_list =pd.DataFrame()
    info_needed = [key for key in degree_dict if not degree_dict[key]]
    all_degree_info = False
    i = edu_start
    in_edu = True
    university_s = ""
    major_s = ""
    while in_edu:
        s = resume_df.Block_Title[i]
        s = re.sub("('|’)s", "s", s)
        s = re.sub("\s{1,}", " ", s)
        s = re.sub("&","and",s)

        date_s, s = get_date_and_remove_it_from_title(s)
        s = s.strip()
        gpa_s, s = extract_GPA_clean_s(s)
        s = s.strip()
        s = re.sub("\s*(W|w)ith$", "", s)
        if date_s:
            if degree_dict["DATE"]:
                edu_list = edu_list.append(degree_dict,ignore_index = True)
                for key in degree_dict:
                    degree_dict[key] = ""
            degree_dict["DATE"] = date_s
        s = re.sub(r'\([^)]*\)', '', s)
        if gpa_s:
            if degree_dict["GPA"]:
                edu_list = edu_list.append(degree_dict,ignore_index = True)
                for key in degree_dict:
                    degree_dict[key] = ""
            degree_dict["GPA"] = gpa_s
        s = s.strip()
        #s = s.replace(".", "")
        #s = re.sub("(MO|MS|MD)$","",s)
        s = re.sub("\A[^a-zA-Z0-9]","",s).strip()
        if s :
            city_s,s = get_city_allennlp_and_remove_it(s)
        # if re.search("\,?\s?[A-Z]{2}$",s):
        #     s = re.sub("\,?\s?[A-Z]{2}$","",s).strip()
        #     #s2 = re.search("[A-Za-z]{4,}$",s)
        #     # city = extract_city(s[s2.start():s2.end()])
        #     # print(city)
        #     # if city:
        #     #     s = rreplace(s,city[0],"",1)
        #     s = get_city_allennlp_and_remove_it(s)
        degree, degree_type = extract_degree_level(s)
        s = re.sub("\s{0}(-|-|—|-)\s?","",s)
        s = re.sub("[^a-zA-Z0-9\s,.]",",",s)
        s = re.sub(",$","",s).strip()
        s = re.sub(",$","",s).strip()
        s = re.sub("\s+,",",",s)
        major_s = ""
        university_s = ""
        if degree_type:
            s = re.sub("(\A[^a-zA-Z0-9]|[^a-zA-Z0-9]$)", "", s)
            s = s.strip()
            p_at = re.search("\s+(at|AT|At)\s+",s)
            p_in = re.search("\s+(In|in|IN)\s+",s)
            if p_at and p_in:
                doc = nlp(s)
                for ent in doc.ents:
                    if ent.label_== "ORG":
                        university_s = ent.text
                    elif ent.label == "MAJ":
                        major_s = ent.text
                if university_s and not major_s:
                    major_s = s[p_in.start():p_in.end()]
            elif not re.search("[^a-zA-Z0-9\s\.]",s) and re.search("\s+(I|i)(N|n)\s+",s):
                tp = re.split("\s+(In|IN|in)\s+", s,1)
                tp = [x.strip() for x in tp if x]
                tp = [x for x in tp if x not in ["in","IN","In"]]
                major_s = tp[1]
            elif re.search("[^a-zA-Z0-9\s\.]",s) and re.search("\s+(I|i)(N|n)\s+",s):
                tp = re.split("([^a-zA-Z0-9\s\.])",s)
                tp = [x for x in tp if x]
                tp = [x for x in tp if not re.search("(\s*[^a-zA-Z0-9\s\.])", x)]

                for p,val in enumerate(tp):
                    pos = re.search("\s+(In|IN|in)\s+",val)
                    if pos:
                        major_s = val[pos.end():]
                        s = s.replace(val,"")
                        degree_pos = p
                        break
                tp.pop(p)
                if len(tp) == 1:
                    university_s = tp[0]
            elif not re.search("[^a-zA-Z0-9\s\.]",s) and not p_in:
                degree = degree.strip()
                if degree[-1] == ".":
                    s = s.replace(degree[:-1], "")
                    s = s.replace(".", "").strip()
                else:
                    s = s.replace(degree,"")
                    s = re.sub("[^a-zA-Z0-9\s]","",s).strip()
                    if s:
                        major_s = s
            else:
                degree = degree.strip()
                if degree[-1] == ".":
                    s = s.replace(degree[:-1], "")
                    s = s.replace(".","").strip()
                doc = nlp(s)
                p = 0
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        if p == 0:
                            university_s = ent.text
                            p += 1
                        else:
                            if not major_s:
                                major_s = university_s
                                university_s = university_s+" "+ent.text
                    elif ent.label_ == "MAJ":
                        if not major_s:
                            major_s = ent.text
                        else:
                            major_s = major_s + " ; "+ent.text
                if not university_s and not major_s and city_s:
                    s = re.sub("[^a-zA-Z0-9\s]", "", s)
                    degree = re.sub("[^a-zA-Z0-9\s]", "", degree)
                    s = s.replace(degree, "")
                    university_s = s



            if university_s:
                if degree_dict["ORG"]:
                    edu_list = edu_list.append(degree_dict, ignore_index=True)
                    for key in degree_dict:
                        degree_dict[key] = ""
                degree_dict["ORG"]= university_s
            if degree_dict["DEGREE"]:
               edu_list = edu_list.append(degree_dict,ignore_index = True)
               for key in degree_dict:
                   degree_dict[key]=""
            degree_dict["DEGREE"] = degree_type
            if major_s:
                if degree_dict["MAJ"]:
                    edu_list = edu_list.append(degree_dict, ignore_index=True)
                    for key in degree_dict:
                        degree_dict[key] = ""
                degree_dict["MAJ"] = major_s



        else:
            #remove everything between brackets
            if degree_dict["ORG"] and degree_dict["DEGREE"] and not degree_dict["MAJ"] and s:
                pos = s.split(":")
                if len(pos) == 2:
                    degree_dict["MAJ"] = pos[1]
                else:
                    degree_dict["MAJ"] = s
            elif not degree_dict["ORG"] and degree_dict["DEGREE"] and degree_dict["MAJ"] and s:
                degree_dict["ORG"] = s
            else:
                doc = nlp(s)
                university_s = ""
                major_s = ""
                for ent in doc.ents:
                    if ent.label_ == "MAJ":
                        if not major_s:
                            major_s = ent.text
                        else:
                            major_s = major_s + " : "+ent.text
                    elif ent.label_ == "ORG":
                        university_s = university_s + " " + ent.text
                if university_s:
                    if degree_dict["ORG"] and not degree_dict["MAJ"]:
                        degree_dict["MAJ"] = university_s
                    elif  degree_dict["ORG"]:
                        edu_list = edu_list.append(degree_dict,ignore_index = True)
                        for key in degree_dict:
                            degree_dict[key] = ""
                    degree_dict["ORG"] = university_s
                if major_s:
                    if degree_dict["MAJ"]:
                        edu_list = edu_list.append(degree_dict, ignore_index=True)
                        for key in degree_dict:
                            degree_dict[key] = ""
                    degree_dict['MAJ']= major_s
        info_needed = [key for key in degree_dict if not degree_dict[key]]
        i += 1
        if i == len(resume_df):
            in_edu = False
        elif check_if_heading(resume_df.Block_Title[i]) :
            edu_list = edu_list.append(degree_dict,ignore_index= True)
            in_edu = False

    return edu_list,i

def clean_title_df(s):
    '''
    clean_title_df: clean string from weird patters
    :param s: string
    :return: cleaned strings
    '''
    pattern1 = "\|\s|(\:)$"
    pattern2 = "^[a-z]\."
    s = s.replace("\t", " ")
    s = s.replace(u'\xa0', u' ')
    s = re.sub(pattern2, "", s)
    s = re.sub(pattern1, "", s)
    s = re.sub("^[:xdigit:]", "", s)
    s = re.sub('\A[^A-Za-z0-9\s\(\+]+', '', s)
    s = s.strip()
    return (s)





def check_if_potential_title(st):
    '''
    check_if_potential_title: check if the string is a title
    :param st: string
    :return: boolean (true if title, else false)
    '''
    s = st.strip()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\([^)]*', '', s).strip()
    s = re.sub("[a-z]\'[a-z]\s+"," ",s)
    s = re.sub("\s+[a-z]{1,2}\s+",", ",s)
    s = re.sub("\s+[^a-zA-Z0-9\s\.\,]\s+",", ",s)
    s = re.sub("\A,","",s).strip()
    if len(s) <= 3:
        return (False)
    if re.search("\s*((C|c)*(GPA|gpa)\:*)", s):
        return (True)
    if re.search("\A[a-z][A-Z]",s):
        return(True)
    if re.search("\A([a-z]|(A\s+))", s):
        return (False)
    if is_email(s):
        return(False)
    if re.search("\A(Skills|SKILLS)", s):
        return (True)
    if re.search("\D\d{3}\D",s):
        return(False)
    tp = s.split()
    tp = [x for x in tp if x]
    tp = [x.strip() for x in tp]
    stop_words = set(stopwords.words('english'))
    if len(tp) >= 2:
        word_tokens = word_tokenize(s)
        filtered_sentence = [w for w in word_tokens if w in stop_words]
        left = set(filtered_sentence) - set(["in", "of", "at", "the", "for", "and", "to", "by", "s"])
        if left:
            return (False)
    s_tp = re.sub(r'[^\w\s]', '', s)
    word_tokens = word_tokenize(s_tp)
    tp = [w for w in word_tokens if w not in stop_words]
    tp = [w for w in tp if w not in ["de","die"]]
    tp = [w for w in tp if not re.search("\A\d{1,}\Z",w)]
    if tp:
        all_cap = [x for x in tp if x.istitle()|x.isupper()]
        if len(all_cap) == len(tp):
            return (True)
        elif len(all_cap) == len(tp)-1:
            return(True)
        else:
            return(False)


    if re.search("\A[^a-zA-Z0-9+]", s):
        return (False)
    if re.search("\A[A-Z]{1}[a-z]{2,}\.$", s):
        return (False)
    if re.search("([A-Z]{3,}|[A-Z][a-z]{3,})\.$", s):
        return (False)
    s = s.replace("\t", " ")
    s = s.replace("&", "")
    s = s.replace("|", "")
    if re.search("\s+(1|2)\d{3}\Z", s):
        return (True)
    if re.search("^(1|2){3}\Z", s):
        return (True)
    if re.search("\A(1|2)\d{3}", s):
        return (True)
    date_s, s = get_date_and_remove_it_from_title(s)
    if re.search("\s+\d{5}$", s):
        return (False)
    if date_s:
        return(False)
#MAY NOT NEED ALL THESE
    match = re.findall("[:,]", s)
    if len(match) > 4:
        return (False)

    s = s.strip()

    if re.search("(\A\(|\)$)", s):
        return (False)
    s = re.sub('[^\w\s]', "", s)
    tp = remove_stopwords(s)
    s = " ".join(tp)
    tp = [x for x in tp if not re.search("[^a-zA-Z]", x)]
    val_p = [x for x in tp if re.search("\A[A-Z]", x)]
    # val_p = [x for x in tp if re.search("[A-Z]+[a-z]+\:*$",x)]
    if not val_p:
        return (False)
    elif len(tp) == len(val_p):
        return (True)
    elif len(val_p) < len(tp):
        return (False)
    else:
        tmp = s.split(" ")
        tmp = [x for x in tmp if not x == '']
        tmp = [x.strip() for x in tmp]
        if (len(tmp) == 1) and tmp[0][0].isupper():
            if re.search("[a-z]+\.$", tmp[0]):
                return (False)
            else:
                return (True)

        if (len(tmp) == 2) and (tmp[0][0].isupper()):
            return (True)
        elif (len(tmp) >= 3):
            tmp = [x for x in tmp if x not in ["of", "and", "for", "in", "at"]]
            if tmp[0].isupper():
                return (True)
            if tmp[0][0].isupper():
                if tmp[1][0].isupper():
                    return (True)
                else:
                    return (False)
            else:
                return (False)
        else:
            return (False)


def multiple_replace(string, reps, re_flags=0):
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
    replacements = [("\t", ""), ("&", ""), ("\/", ""), (":", ""), (",", ""), ("-", " "), (" for ", " "),
                    ("–", ""), (" at ", " "), (" of ", " "), (" in ", " "), (" to ", " "), (" and ", " "),
                    ("[0-9]", "")]
    s = multiple_replace(s, replacements)
    tmp = s.split(" ")
    tmp = [x.strip() for x in tmp if x != " "]
    tmp = [x for x in tmp if x]
    upper_val = [x for x in tmp if x[0].isupper()]
    if (len(tmp) <= 2 & len(upper_val) == 1):
        return (True)
    elif len(tmp) == len(upper_val):
        return (True)
    else:
        return (False)


'''
gather_headings:  get all the headings (titles) from the resume.
Paramerters:
  doc_list:  reading pdf to text result in a text with return lines, so a split result in a list of strings
  resume_name: the file name of the resume
  
Returns:
  blocks_df: a pandas df
'''


# def gather_headings(doc_list, resume_name):
#     blocks_df = pd.DataFrame()
#     for i, block in enumerate(doc_list):
#         block = clean_title_df(block)
#         tmp = block.split()
#         if len(tmp) == 1:
#             block = block.strip()
#             if block[0].isupper():
#                 tmp_df = pd.DataFrame({"Resume_Name": resume_name, "Block_Pos": i, "Block_Title": block}, index=[i])
#                 blocks_df = blocks_df.append(tmp_df)
#         elif check_if_potential_title(block):
#             tmp_df = pd.DataFrame({"Resume_Name": resume_name, "Block_Pos": i, "Block_Title": block}, index=[i])
#             blocks_df = blocks_df.append(tmp_df)
#     blocks_df.reset_index()
#     tp = [x for ix, x in blocks_df.iterrows() if check_number_uppercases_words(x.Block_Title)]
#     blocks_df = pd.DataFrame(tp)
#     # blocks_df.reset_index()
#     return (blocks_df)


# In[292]:


def V2_gather_headings(doc_list, resume_name):
    blocks_df = pd.DataFrame()
    pattern2ignore = "\A([A-Z][a-z]{3,}\s*){2,3}\."
    for i, block in enumerate(doc_list):
        block = clean_title_df(block)
        tmp = block.split()
        if check_if_potential_title(block):
            tmp_df = pd.DataFrame({"Resume_Name": resume_name, "Block_Pos": i, "Block_Title": block}, index=[i])
            blocks_df = blocks_df.append(tmp_df)
            # blocks_df.reset_index()
    # tp = [ x for ix, x in blocks_df.iterrows() if check_number_uppercases_words(x.Block_Title)]
    # blocks_df = pd.DataFrame(tp)
    # blocks_df.reset_index()
    return (blocks_df)



def maybe_company(s):
    m = s.split()
    if m[0].isupper():
        return (True)
    else:
        return (False)


def get_date_or_company(st):
    date_s = ""
    company_s = ""
    s = st.strip()
    doc = nlp(s)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_s = ent.text
            if not date_s:
                _, end, date_s = get_date(s)
        elif ent.label_ == "ORG":
            company_s = ent.text
    return (date_s, company_s)


def extract_degree_level(st):
    '''
    extract_degree_level: get the degree level (Bachelor, Master, Doctor) and degree type BS, MSc.
    :param st: string
    :return: degree level and degree type
    '''
    s = st.strip()
    s = re.sub("[^a-zA-Z0-9\s]",".",s)
    doc = nlp(s)
    degree = ""
    degree_type = ""
    major_s = ""
    for ent in doc.ents:
        if ent.label_ == "EDU":
            degree_type = ent.ent_id_
            degree = ent.text
    return degree, degree_type


def check_if_city_and_remove_it(s):
    '''
    check if city and remove it: using spacy, find the city and remove it
    :param s: string
    :return: string with out the city
    '''
    doc = nlp(s)
    n = 1
    for ent in doc.ents:
        if ent.label_ == "GPE":
            s = rreplace(s, ent.text, '', 1).strip()
            s = re.sub("\,$", "", s).strip()
            return s
    return ""

def extract_job_experience_text(resume_l,start_ndx,end_ndx):
    '''
    extract_city_experience_text: from resume_l, extract all texts between start index and end index
    :param resume_l: list
    :param start_ndx: start position
    :param end_ndx: end position
    :return: experience text
    '''
    text_exp = ""
    for text_n in range(start_ndx,end_ndx):
        text_exp = text_exp+ "\n"+resume_l[text_n]
    return text_exp

# GPA re.search("GPA\:*\s+[0-9]{1,2}\.*([0-9]{1,2})*","Master of Technology in Intelligent Systems, GPA 4")
def extract_GPA_clean_s(st):
    pos = re.search("\(?(C)?GPA[^a-zA-Z\d]?\s*[0-9]{1,2}\.*([0-9]{1,2})*\s?(\/\s?([0-9]{1,2})?\.?([0-9]{1,2})?\)?)?",
                    st)
    if pos:
        s_gpa = st[pos.start():pos.end()]
        st = st.replace(s_gpa, "")
    else:
        pos2 = re.search("(\s+\d{1,2}\.?(\d{1,2})?\s+(C)?GPA|GPA\s*\d{1,2}\.?([0-9]{1,2})?)", st)
        if pos2:
            s_gpa = st[pos2.start():pos2.end()]
            st = st.replace(s_gpa, "")
        else:
            pos3 = re.search("Overall Percentage:\s*\d{2}(.\d{1,2})?",st)
            if pos3:
                s_gpa = st[pos3.start():pos3.end()]
                st = st.replace(s_gpa, "")
            else:
                s_gpa = ""
    return (s_gpa, st)


def extract_degree_IN(st):
    doc = nlp(st)
    for ent in doc.ents:
        print(ent.label_, ent.text)
    return ("1")


def extract_all_info_degree(s):
    degree_dict = {"GPE": "", "ORG": "", "Bachelor": "", "Master": "", "Doctor": "", "Major": "", "DATE": ""}
    degree_dict["DATE"], s = get_date_and_remove_it_from_title(s)
    s = re.sub("\s+(c|C)um\s+(l|L)aude", "", s)
    s = re.sub("\s+(S|s)uma\s+", "", s)
    s = re.sub("(m|M)agna", "", s)
    if re.search("\s(A|a)t\s", s):
        tp = re.split("\s+(A|a)t\s+", s)
        l = len(tp) - 1
        degree_dict["ORG"] = tp[l]
        s = tp[0]
    doc = nlp(s)
    degree_c = 1
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if not degree_dict["DATE"]:
                degree_dict[ent.label_] = ent.text
                s = s.replace(ent.text, "")

        else:
            if ent.label_ == "EDU":
                degree_dict[ent.ent_id_] = ent.text
                s = s.replace(ent.text, "")
            elif ent.label_ == "ORG":
                if not degree_dict["ORG"]:
                    degree_dict["ORG"] = degree_dict["ORG"] + " | " + ent.text
                    s = s.replace(ent.text, "")
            elif ent.label_ == "GPE":
                if not degree_dict["GPE"]:
                    degree_dict["GPE"] = ent.text
                    s = s.replace(ent.text, "")

            else:
                if ent.label_ in list(degree_dict.keys()):
                    degree_dict[ent.label_] = ent.text
                    s = s.replace(ent.text, "")
    if not degree_dict["DATE"]:
        tp = re.search("\s\d{4}$", s)
        if tp:
            degree_dict["DATE"] = s[tp.start():]
            s = s[:tp.start()]

    if not degree_dict["GPE"]:
        degree_dict["GPE"] = extract_city(s)
        if degree_dict["GPE"]:
            l = len(degree_dict["GPE"])
            degree_dict["GPE"] = degree_dict["GPE"][l - 1]
            s = s.replace(degree_dict["GPE"], "")
        else:
            degree_dict["GPE"] = ""
    if not degree_dict["Major"]:
        s = s.replace(".", "")
        s = re.sub("\,", "", s)
        s = s.strip()
        if re.search("School|University|College|Institute", s):
            degree_dict["Major"] = degree_dict["ORG"]
            degree_dict["ORG"] = re.sub("[^a-zA-Z&\s]", "", s).strip()
        else:
            tp = re.split("(IN|in|In)", s)
            if len(tp) > 1:
                degree_dict["Major"] = re.sub("[^[a-zA-Z&\s]", "", tp[len(tp) - 1]).strip()
            else:
                degree_dict["Major"] = s

    return degree_dict


'''def get_degree_university_date(s):
doc = nlp("South Dakota State University, Brookings MA, Economics 2014")
for ent in doc.ents:
    print(ent.label_,ent.text)'''


# city_s = extract_city("Augustana University, Sioux Falls BA, Econ & Math 2012")
# print(major, " * ",city_s," * ",degree, " * ", company_s, " * ",date_s)
# s = "Fordham University, New York City PhD, Economics In progress, ABD"
# extract_city("New York City")
def get_job_info2(s):
    found = False
    found_date = False
    job_dict = {"DATE": "", "ORG": "", "GPE": "", "Job_Title": ""}
    job_dict["GPE"] = extract_city(s)
    if job_dict["GPE"]:
        l = len(job_dict["GPE"]) - 1
        job_dict["GPE"] = job_dict["GPE"][l]
        s = s.replace(job_dict["GPE"], "")
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
        print("s ", s)
        if ent.label_ == "GPE":
            if not job_dict["GPE"]:
                job_dict["GPE"] = ent.text
                s = s.replace(ent.text, "")
        elif ent.label_ == "DATE":
            if n == 1:
                job_dict["DATE"] = ent.text
                s = s.replace(ent.text, "")
                n += 1
            else:
                job_dict["DATE"] = job_dict["DATE"] + " - " + ent.text
                s = s.replace(ent.text, "")
            found_date = True
        elif ent.label_ == "ORG":
            job_dict[ent.label_] = job_dict[ent.label_] + " : " + ent.text
            s = s.replace(ent.text, "")

    if not job_dict["DATE"]:
        job_dict["DATE"], s = get_date_and_remove_it_from_title(s)

    if not found:
        s = s.replace(".", "")
        s = re.sub("[A-Z]{2}", "", s)
        s = re.sub("[^a-zA-Z\s]", "", s)
        job_dict["Job_Title"] = s.strip()
    if isinstance(job_dict["GPE"], list):
        job_dict["GPE"] = ""
    return job_dict


# In[873]:


def extract_city(s):
    places = geograpy3.get_place_context(text=s)
    return (places.cities)


# In[16]:


def count_n_POS(s):
    tmp = s.split(",")
    tp = nltk.pos_tag(nltk.word_tokenize(tmp[0]))
    res = [val for i, val in tp]
    res = set(res)
    return (res)


def is_phone_number(s):
    res = re.sub(r'[^\w^\s]', '', s)
    pos = re.search("\d{10,}", s)
    if pos:
        return(True)
    else:
        return(False)




# "from find_job_titles import FinderAcora"

# s = 'SR. DATA PROCESSING SYSTEMS ANALYST'
# Find Job Title
def get_job_title(s):
    f = re.search("(QA\s|Head|HEAD|ASSISTANT|Intern|INTERN|Assistant|ANALYST|\
    Analyst|PRINCIPAL|Principal|Consultant|CONSULTANT)", s)
    if f:
        start, end = f.span(0)
        title_s = f.string
        print(start, end)
        yield start, end, title_s[start:end]
    else:
        s = s.title()
        finder = FinderAcora()
        result = finder.finditer(s)
        try:
            gen = iter(result)
            it = next(gen)
            start = it.start
            end = it.end
            title_s = it.match
            yield start, end, title_s
        except RuntimeError:
            # return start, end, title_s
            yield 0, 0, ""


# In[193]:


# Teach For America – High School STEM Teacher, Chicago, IL May 2011-December 2012
# Or BOSTON POLICE DEPARTMENT – Boston, MA 2001 TO 2007
'''Pattern if - exist.  In this case, we can have 2 possibilities, either 
the first value is the company or the second is the company'''

'''If date ad city found than we can assume that we have the company ad the job title
'''


def get_city_and_remove_it(st):
    city_s = extract_city(st)
    if city_s:
        city_s = city_s[len(city_s) - 1]
        matches = re.finditer(city_s, st)
        matches_positions = [match.start() for match in matches]
        matches_positions = matches_positions[len(matches_positions) - 1]
        s = st
        s = s[:matches_positions - 1]
        print("Company ", s)
        return city_s, s
    else:
        return "", st


# In[458]:


# In here we assume city and company after job description
# but we have Company, City, KY. - Title

'''
CHECK IT OUT 
'''


def get_title_company_AT(st, pattern_at):
    tp = re.split(pattern_at, st)
    tp = [x.strip() for x in tp if x]
    tp = [x for x in tp if x not in ["at", "At"]]
    job_title = tp[0]
    city_s = extract_city(tp[1])
    if city_s and re.search("\,", tp[1]):
        tp_city = re.split("\,", tp[1])
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


def get_work_heading(st, punct):
    s = st.strip()
    date_s, s = get_date_and_remove_it_from_title(s)
    pattern_at = "\s+(at|At|AT)\s+"
    pattern_comma = "\s*(\,|\||/)\s+"
    pattern_under = "\s*(-|—\|)\s*"
    if re.search(pattern_at, s):
        job_title, company_s = get_title_company_AT(s, pattern_at)
    elif re.search(pattern_under, s):
        tp = re.split(punct, s)
        tp = [x.strip() for x in tp if x]
        tp = [x for x in tp if x not in ["-", "—", "|"]]
        print(tp)
        # print(len(tp),"tp " ,tp)
        if len(tp) == 3:
            tp[1] = tp[len(tp) - 1]
        city_s, company_s = get_city_and_remove_it(tp[1])
        if not city_s:
            city_s, company_s = get_city_and_remove_it(tp[0])
            job_title = tp[1]
        if city_s:
            job_title = tp[0]


    else:
        tp = re.split(pattern_comma, s)
        print(tp)
        tp = [x.strip() for x in tp if x]
        tp = [x for x in tp if x not in [",", "|", "/"]]
        start, end, job_title = list(get_job_title(tp[0]))[0]
        if end == 0:
            job_title = tp[1]
            company_s = tp[0]
        else:
            job_title = tp[0]
            company_s = tp[1]
    job_dict = {"job Title": job_title, "Company": company_s, "Dates": date_s}
    return (job_dict)


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
        res = get_work_heading(s, pattern1)
        job_title = res["job Title"]
        company_s = res["Company"]
    else:
        matches = re.finditer(",|\|", s)
        matches_positions = [match.start() for match in matches]

        if matches_positions:
            tp = re.split("(,|\|)", s)
            tp = [x for x in tp if x not in [",", "|"]]
            print(" SSSS ", s, len(tp))
            print(" TPP ", tp)
            if len(tp) == 2:
                _, company_s = get_date_or_company(tp[0])
                if company_s:
                    start, end, job_title = list(get_job_title(tp[1]))[0]
                else:
                    start, end, job_title = list(get_job_title(tp[0]))[0]
                    _, company_s = get_city_and_remove_it(tp[1])



        else:
            start, end, job_title = list(get_job_title(s))[0]
            if end == 0:
                company_s = s

    job_dict = {"job Title": job_title, "Company": company_s, "Dates": date_s}
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
        return (True)


def is_github_or_linkedIn(s):
    pattern2 = '(g|G)ithub|(l|L)inked(i|I)n'
    if any(re.findall(pattern2, s, re.IGNORECASE)):
        return (True)
    else:
        return (False)


def check_for_nns_nnps(st):
    text = st.title()
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for val, pos_v in tag:
        if pos_v in ["NNPS", "NNS"]:
            return (True)
    return (False)


# In[199]:
def check_if_heading(st):
    if not isinstance(st, str):
        return ("")
    s = st.strip()
    s = s.replace("/", " ")
    s = s.lower()
    if re.search("\Aeducation$", s):
        return ("EDU")
    elif re.search("\A(education\s*(and|\,|&)*)", s):
        return ("EDU")
    elif re.search("\Alanguage(s)*(?<!:)$", s):
        return ("SKI")
    elif re.search("\A(language(s)*\s*(and|\,|&)*)", s) \
            or re.search("\A[a-z]*\s+(skills)", s) \
            or re.search("\A(skills$|skills\s+(and|\,|&)*)", s):
        return ("SKI")
    elif re.search("\A(projects$|projects\s+(and|\,|&)*)", s) or \
            re.search("\A[a-z]*\s+(projects)", s):
        return ("PRO")
    elif re.search("summary", s):
        return ("SUM")
    else:
        if s in heading_dict:
            return (heading_dict[s])
        else:
            return ("")


def find_heading_position(resume_df, chosen_heading):
    tmp_l = [x.strip() for x in chosen_heading]
    for i, item in enumerate(resume_df.Block_Title):
        if str(item).lower() in tmp_l:
            return (int(i))
    return (-1)


# In[200]:





# Detect start of an experience (Company name, location, position, place)

def V2_detect_start_experience(text_l, pos, n):
    global keep_Experience_headings
    s = text_l[pos]
    s = s.replace("\t", ' ')
    i = pos
    not_start = True
    while not_start:
        _, _, date_s = get_date(s)
        if (not is_symbol(s) and (len(count_n_POS(s)) < 4) and i < n) or check_if_potential_title(s):
            if (i >= len(text_l)):
                not_start = False
            else:
                keep_Experience_headings = keep_Experience_headings.append(pd.DataFrame({"Exp_Heading": s}, index=[0]),
                                                                           ignore_index=True)
                s = text_l[i]
                s = s.replace("\t", ' ')

            i += 1
        elif (i >= n):
            not_start = False
        elif (len(count_n_POS(s)) >= 4):
            not_start = False

    return (i)


# In[342]:


def V3_detect_start_experience(text_l, pos, n):
    s = text_l[pos]
    pos_l = pos
    start_pos = next(iter(resume_df[resume_df['Block_Pos'] == pos_l - 1].index), -1)
    while start_pos:
        pos_l += 1
        start_pos = next(iter(resume_df[resume_df['Block_Pos'] == pos_l - 1].index), -1)
        if pos_l >= n:
            start_pos = False
    return (pos_l - 1)


# In[202]:


# Extract the paragraph or block with the experience

def extract_experience(text_l, pos, n):
    i = pos
    not_end = True
    experience_txt = ""
    # Is lower, and number of words is more than
    while not_end:
        if i >= n:
            not_end = False
        else:
            s = text_l[i]
            s = s.replace("\t", ' ')
            if check_if_potential_title(s):
                not_end = False
            else:
                if is_github_or_linkedIn(s) or is_email(s) or is_phone_number(s):
                    i = n
                    not_end = False
                else:
                    experience_txt = experience_txt + "\n" + s
            i += 1
    return (experience_txt, i - 1)


# In[203]:


def get_all_experiences(resume_l, start_t, end_extract_pos):
    experience_df = pd.DataFrame()
    keep_extracting_exp = True
    len_resume = end_extract_pos
    start_pos = start_t
    counter = 1
    while keep_extracting_exp:
        start_exp = V2_detect_start_experience(resume_l, start_pos, len_resume)
        if start_exp == len_resume - 1:
            keep_extracting_exp = False
        # break
        else:
            exp_txt, end_exp = extract_experience(resume_l, start_exp - 1, len_resume)
            start_pos = end_exp
            experience_df = experience_df.append({"Experiences": exp_txt, "Counter": counter}, ignore_index=True)
            counter += 1
            if end_exp >= len_resume - 1:
                keep_extracting_exp = False

    return (experience_df)


# In[ ]:




def parse_all_engineers_resume():
    path_name = "../HSS_Resumes/"
    resume_name = "Engineers Resume_df201026.csv"
    csv_filename = path_name + resume_name
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

def get_job_info_1_resume(pdf_document):
    global job_list,master_job_df,resume_df,resume_l,master_ski_df,master_edu_df
    resume_df = pd.DataFrame()
    resume_l = []
    text = read_pdf_resume(pdf_document)
    # text = open(pdf_document, "r").read()
    resume_l = initial_cleaning(text)
    resume_df = V2_gather_headings(resume_l, resume_name)
    resume_df.reset_index(inplace=True, drop=True)
    candidate_name = get_candidate_name(resume_l)
    if candidate_name:
        resume_df = resume_df[resume_df.Block_Title != candidate_name]
        resume_df.reset_index(inplace=True, drop=True)
    found = False
    counter = 0
    print(resume_name)
    job_list = pd.DataFrame()
    while not found:
        if check_if_heading(resume_df.Block_Title[counter]) == "TXT":
            job_list = get_job_info(resume_df, resume_l, counter + 1)
            master_job_df = master_job_df.append(job_list, ignore_index=True)
            master_job_df.to_csv("hiring_manager_Job_201118-24-12.csv")
            found = True
        elif check_if_heading(resume_df.Block_Title[counter]) == "XXX":
            skill_df,i = get_skills_info(resume_df,resume_l,counter)
            master_ski_df = master_ski_df.append(skill_df)
            master_ski_df.to_csv("hiring_manager_ski_201119-2-29.csv")
            found = True
        elif check_if_heading(resume_df.Block_Title[counter]) == "EDU":
            edu_list,i = get_education_info(resume_df, counter+1)
            master_edu_df = master_edu_df.append(edu_list)
            master_edu_df.to_csv("hiring_manager_edu_201119-2-29.csv")
        counter += 1
        if counter == resume_df.shape[0]:
            found = True
    return master_ski_df
# In[1008]:
nlp = spacy.load('./')
initialize_headings_file()
resume_2_process = pd.read_csv("../Resume2Parse.csv")
path_name = "../HSS Resumes/hiring-manager/"
for j in range(2,29):
    resume_name = "hiring_manager_review1 (dragged) "+str(j)+".pdf"
    pdf_document = path_name + resume_name
    final_df = get_job_info_1_resume(pdf_document)
final_df.reset_index(inplace=True, drop=True)
final_df.to_csv("resume_hiring_manager_review1_edu_201119.csv")


