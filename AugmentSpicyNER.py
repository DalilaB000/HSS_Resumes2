import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import pandas as pd

def augment_spacy_ner():
    degrees_df = pd.read_csv("../Degrees And Acronyms.csv")
    major_df = pd.read_csv("../Majors_Look_Up.csv")
    nlp = spacy.load("en_core_web_md")
    ruler = EntityRuler(nlp)
    for row in degrees_df.itertuples():
        ruler.add_patterns([{"label": row.Type, "pattern": row.DegreeAcronym,"id":row.DegreeName}])
    for row in major_df.itertuples():
        ruler.add_patterns([{"label": row.Entity, "pattern": row.Majors}])
    nlp.add_pipe(ruler,before ="ner")
    nlp.to_disk("./")

augment_spacy_ner()