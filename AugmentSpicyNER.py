import pandas as pd
import spacy
from allennlp.predictors.predictor import Predictor
from spacy.pipeline import EntityRuler
import pytesseract as pt
import pdf2image
import allennlp_models.tagging


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
def convert_pdf_to_txt(path_name):
    pages = pdf2image.convert_from_path(pdf_path=path_name, dpi=200, size=(1654, 2340))
    content = ""
    for i in range(len(pages)):
        content = content+"\n"+pt.image_to_string(pages[1])
    return content
#augment_spacy_ner()
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz")
tp = predictor.predict(
    sentence="Obelisk Solutions"
)
for word,tag in zip(tp["words"],tp["tags"]):
   print(f"{word}\t{tag}")
