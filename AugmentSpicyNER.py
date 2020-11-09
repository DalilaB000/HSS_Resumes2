import pandas as pd
import spacy
from allennlp.predictors.predictor import Predictor
from spacy.pipeline import EntityRuler
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

#augment_spacy_ner()
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz")
tp = predictor.predict(
  sentence="Vizuro – Remote work, USA Nov 2018 – Present"
)
for word,tag in zip(tp["words"],tp["tags"]):
    print(f"{word}\t{tag}")
