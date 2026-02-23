import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dtsc330_26.readers.articles import Articles
from dtsc330_26.readers.grants import Grants

articles = Articles(str(PROJECT_ROOT / "data/pubmed26n1335.xml.gz"))
grants = Grants(str(PROJECT_ROOT / "data/RePORTER_PRJ_C_FY2025.zip"))

article_authors = articles.get_authors()
grant_grantees = grants.get_grantees()
grant_main = grants.get_grants()

article_people = article_authors.rename(columns={
    "PMID": "id",
    "ForeName": "forename",
    "LastName": "surname",
    "Initials": "initials",
    "Affiliation": "affiliation",
})[["id", "forename", "surname", "initials", "affiliation"]].copy()

grant_people = grant_grantees.merge(
    grant_main[["application_id", "organization"]],
    on="application_id",
    how="left",
)
grant_people["forename"] = grant_people["pi_name"].str.split(",").str[-1].str.split().str[0]
grant_people["surname"] = grant_people["pi_name"].str.split(",").str[0].str.strip()
grant_people["initials"] = grant_people["forename"].str[0].str.upper()
grant_people["affiliation"] = grant_people["organization"]
grant_people["id"] = grant_people["application_id"]

grant_people = grant_people[["id", "forename", "surname", "initials", "affiliation"]].copy()

print(article_people.head(3))
print(grant_people.head(3))

#clean names
article_people["forename"] = article_people["forename"].astype(str).str.strip()
article_people["surname"] = article_people["surname"].astype(str).str.strip()

grant_people["forename"] = grant_people["forename"].astype(str).str.strip()
grant_people["surname"] = grant_people["surname"].astype(str).str.strip()

article_people["surname_key"] = article_people["surname"].str.lower()
grant_people["surname_key"] = grant_people["surname"].str.lower()

article_people["first_letter"] = article_people["forename"].str[0].str.lower()
grant_people["first_letter"] = grant_people["forename"].str[0].str.lower()

# matches based on surname and first letter of forename
same_person_pairs = article_people.merge(grant_people,on=["surname_key", "first_letter"],suffixes=("_article", "_grant"))

same_person_pairs = same_person_pairs.sample(n=min(10, len(same_person_pairs)),random_state=42).copy()

same_person_pairs["label"] = 1
same_person_pairs["error_style"] = "none"

# non matches based on surname and first letter of forename
random_articles = article_people.sample(n=200, replace=True, random_state=42).reset_index(drop=True)
random_grants = grant_people.sample(n=200, replace=True, random_state=43).reset_index(drop=True)

different_person_pairs = pd.concat([random_articles.add_suffix("_article"), random_grants.add_suffix("_grant")],axis=1)

different_person_pairs = different_person_pairs[
    (different_person_pairs["surname_key_article"] != different_person_pairs["surname_key_grant"]) |
    (different_person_pairs["first_letter_article"] != different_person_pairs["first_letter_grant"])
].copy()

different_person_pairs = different_person_pairs.sample(n=min(10, len(different_person_pairs)),random_state=42).copy()

different_person_pairs["label"] = 0
different_person_pairs["error_style"] = "random_nonmatch"


# combine matches and non-matches into a single training set
training_rows = pd.concat([same_person_pairs, different_person_pairs], ignore_index=True)

final_hw5 = training_rows[[
    "forename_article", 
    "surname_article", 
    "initials_article", 
    "affiliation_article",
    "forename_grant", 
    "surname_grant", 
    "initials_grant", 
    "affiliation_grant",
    "label", 
    "error_style"
]].rename(columns={
    "forename_article": "article_forename",
    "surname_article": "article_surname",
    "initials_article": "article_initials",
    "affiliation_article": "article_affiliation",
    "forename_grant": "grant_forename",
    "surname_grant": "grant_surname",
    "initials_grant": "grant_initials",
    "affiliation_grant": "grant_affiliation",
})

print("\nFinal 20-row training sample:")
print(final_hw5.head(20))

# error types
match_rows = final_hw5[final_hw5["label"] == 1].index.tolist()

def add_typo(word):
    if not isinstance(word, str) or len(word) < 3:
        return word
    #  swap 2 middle characters
    chars = list(word)
    i = 1
    if i + 1 < len(chars):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)

# 1) forename typo
if len(match_rows) > 0:
    r = match_rows[0]
    final_hw5.loc[r, "grant_forename"] = add_typo(final_hw5.loc[r, "grant_forename"])
    final_hw5.loc[r, "error_style"] = "forename_typo"

# 2) surname typo
if len(match_rows) > 1:
    r = match_rows[1]
    final_hw5.loc[r, "grant_surname"] = add_typo(final_hw5.loc[r, "grant_surname"])
    final_hw5.loc[r, "error_style"] = "surname_typo"

# 3) initial only
if len(match_rows) > 2:
    r = match_rows[2]
    name = final_hw5.loc[r, "grant_forename"]
    if isinstance(name, str) and len(name) > 0:
        final_hw5.loc[r, "grant_forename"] = name[0]
        final_hw5.loc[r, "error_style"] = "initial_only"

# 4) affiliation abbreviation
if len(match_rows) > 3:
    r = match_rows[3]
    aff = final_hw5.loc[r, "grant_affiliation"]
    if isinstance(aff, str):
        aff = aff.replace("University", "Univ")
        aff = aff.replace("Institute", "Inst")
        aff = aff.replace("Department", "Dept")
        aff = aff.replace("Center", "Ctr")
        final_hw5.loc[r, "grant_affiliation"] = aff
        final_hw5.loc[r, "error_style"] = "affil_abbrev"


print("\nFinal 20-row training sample:")
print(final_hw5.head(20))