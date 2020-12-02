#!/usr/bin/env python
# coding: utf-8

# In[1]:
from typing import Union

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
    global heading_dictionary
    global heading_dict
    heading_dictionary = pd.read_csv("../Lower_big_headings_dictionary.csv")
    heading_dictionary.drop_duplicates(inplace=True)
    heading_dict = heading_dictionary[["Block_Title", "label_id"]]
    heading_dict = dict(heading_dict.values.tolist())





'''
read_pdf_resume: get the path and the name of the pdf resume and read all the pages into a string variable
Parameter:  pdf_document: a full path to a pdf resume
Return: text: all the text in all pages of the pdf document
This was the initial function to reading directly pdf files into a text file; unfortunately, 
all the pdf converters to text libraries has issue with pdf with special fonts, ot type
'''



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
gather_headings:  get all the headings (titles) from the resume.
Paramerters:
  doc_list:  reading pdf to text result in a text with return lines, so a split result in a list of strings
  resume_name: the file name of the resume
  
Returns:
  blocks_df: a pandas df
'''


def V2_gather_headings(doc_list, resume_name):
    blocks_df = pd.DataFrame()
    pattern2ignore = "\A([A-Z][a-z]{3,}\s*){2,3}\."
    for i, block in enumerate(doc_list):
        block = clean_title_df(block)
        tmp = block.split()
        if check_if_potential_title(block):
            tmp_df = pd.DataFrame({"Resume_Name": resume_name, "Block_Pos": i, "Block_Title": block}, index=[i])
            blocks_df = blocks_df.append(tmp_df)
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




def extract_GS(st):
    '''
    extract_GS: find and extract GS level and series from a string
    :param st: string
    :return: grade_level, and grade series
    Note: if multiple grades are in a string, only the latest is kept
    '''
    s = st.strip()
    pattern1 = "\-0\d{3}\-\d{1,2}"
    grades = re.findall(pattern1, s)
    if grades:
        highest_grade = 0
        for grade in grades:
            gd = grade.strip()
            gd = re.sub("\s*", "", gd)
            grade_level = gd[-2:]
            if int(grade_level) > int(highest_grade):
                highest_grade = grade_level
                grade_series = re.search("\d{4}", gd).group()
        grade_level = highest_grade
    else:
        grade_series = re.search("Series:\s?(GS-)?\d{4}",s)
        grade_level = re.search("Grade(\s\w{3,})?:\s?\d{1,2}",s)
        if grade_series and grade_level:
            grade_series = grade_series.string[grade_series.start():grade_series.end()].strip()
            grade_series = grade_series[-4:]
            grade_level = grade_level.string[grade_level.start():grade_level.end()].strip()
            grade_level = grade_level[-2:]
        elif grade_series:
            grade_level = ""
            grade_series = grade_series.string[grade_series.start():grade_series.end()].strip()
            grade_series = grade_series[-4:]
        elif grade_level:
            grade_level = grade_level.string[grade_level.start():grade_level.end()].strip()
            grade_level = grade_level[-2:]
            grade_series = ""

    return grade_level,grade_series

def extract_salary(st):
    '''
    extract_salary: extract the salary from resume
    :param st: string
    :return: salary as string
    the extract patterns  XX,XXX.XX,  XX XXX.XX,  XXXXX.XX, XXX,XXX.XX with either $ in front or USD at the
    end of the amount
    '''
    s = st.strip()
    pattern1 = "((\$\s?\d{1,3}(\,|\s)?\d{3}(\.\d{2})?)|(\s?\d{1,3}(\,|\s)?\d{3}(\.\d{2})?\s*USD))"

    full_salary_pattern = pattern1+"(\s?-\s?"+ pattern1+")?"
    salary = re.search(pattern1,s)
    if salary:
        salary = salary.group().strip()
        return salary
    else:
        return ""

def get_gs_level(text_l):
    found = False
    pos = 0
    val = 0
    grade_series = ""
    grade_level = ""
    for count,text_s in enumerate(text_l):
        text_s = text_s.strip()
        text_s = re.sub("\s+"," ",text_s)
        g_level, g_series = extract_GS(text_s)
        if g_level and g_series:
            grade_level = g_level
            grade_series = g_series
            val = 2
        elif g_level:
            grade_level = g_level
            val += 1
            print("grade ",val)
        elif g_series:
            grade_series = g_series
            val += 1
            print("series ",val)
        if val == 2:
            break
    return grade_level, grade_series








'''def get_degree_university_date(s):
doc = nlp("South Dakota State University, Brookings MA, Economics 2014")
for ent in doc.ents:
    print(ent.label_,ent.text)'''



def extract_city(s):
    places = geograpy3.get_place_context(text=s)
    city = ", ".join(places.cities)
    return (city)

def extract_city_spacy(s):
    doc = nlp(s)
    states = '\s*(AZ|BC|AL|AK|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|LA|ME|MD|MA|MI|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)([^a-zA-Z0-9]|\Z)'
    city = ""
    for ent in doc.ents:
        tp = re.search(states, ent.text)
        if ent.label_ == "GPE":
            if not city:
                city = ent.text
            else:
                city = city + ", " + ent.text
        if tp:
            city = city + ", " + tp.group()
    return city
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






'''RETHINK THIS'''




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



def get_gs_salary(resume_df,resume_l,work_start):
    '''
    get_job_info: extract company name, job title, date, location, and experience (text paragraph)
    :param resume_df: dataframe with titles and their location in resume_l
    :param resume_l: list of lines (strings) in resume
    :param work_start: starting position of job desc in resume_df
    :return: dataframe, and position reached in resume_df
    '''
    job_list = pd.DataFrame()
    job_dict = {"Counter": 0,"Salary":"","Level":"","Series":"","GPE":"","Date" :""}
    #work_start = 3
    in_job = True
    i = work_start
    last_pos =0
    assume_job = False
    counter = 1
    while in_job:
        salary_s = ""
        level_s =""
        series_s = ""
        city_s = ""
        s = resume_df.Block_Title[i]
        s = re.sub("('|’)(s|i)", "s", s)
        s = re.sub("\s{1,}", " ", s)
        s = re.sub("\s+[b-hj-z]\s+",", ",s)
        s = re.sub("\s+gg\s+",",",s)
        g_level, g_grade = extract_GS(s)
        if g_level and g_grade:
            if not job_dict["Level"] and not job_dict["Series"]:
                job_dict["Counter"] = counter
                job_dict["Level"] = g_level
                job_dict["Series"]= g_grade
                counter += 1
        elif g_level:
            if not job_dict["Level"]:
                job_dict["Level"] = g_level
            else:
                job_list = job_list.append(job_dict, ignore_index=True)
                for key in job_dict:
                    job_dict[key] = ""
                job_dict["Counter"] = counter
                job_dict["Level"] = g_level
                counter += 1
        elif g_grade:
            if not job_dict["Series"]:
                job_dict["Series"]= g_grade
            else:
                job_list = job_list.append(job_dict, ignore_index=True)
                for key in job_dict:
                    job_dict[key] = ""
                job_dict["Counter"] = counter
                job_dict["Series"] = g_grade
                counter += 1
        salary_s = extract_salary(s)
        if salary_s:
            job_dict["Salary"]= salary_s
        s = re.sub(r'\([^)]*\)', '', s).strip()
        s = re.sub(r'\([^)]*', '', s).strip()
        date_s, s = get_date_and_remove_it_from_title(s)
        if date_s:
            job_dict["Date"] = date_s
        if len(s) > 2:
            city_s = extract_city_spacy(s)
            if city_s:
                job_dict["GPE"]= city_s
        i += 1
        if i == resume_df.shape[0]:
            job_list = job_list.append(job_dict, ignore_index=True)
            job_list["Resume_Name"] = resume_df.Resume_Name[work_start]
            in_job = False
        elif check_if_heading(resume_df.Block_Title[i]):
            print(last_pos)
            print(resume_df.Block_Pos[i])
            job_list = job_list.append(job_dict, ignore_index=True)
            job_list["Resume_Name"]= resume_df.Resume_Name[i]
            in_job = False

    return job_list

def get_education_info(resume_df, edu_start):
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
        if check_if_heading(resume_df.Block_Title[counter]) == "EXP":
            job_list, i = get_salary_gs(resume_df, resume_l, counter)
            master_job_df = master_job_df.append(job_list, ignore_index=True)
            master_job_df.to_csv("hiring_manager_Job_201118-24-12.csv")
            found = True
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


