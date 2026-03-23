from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import create_engine

sys.path.append(str(Path(__file__).resolve().parents[1]))
from readers.articles import Articles


def insert_grants(engine, data_dir):
    base_dir = Path(__file__).resolve().parents[2]

    csv_path = base_dir / "data" / "RePORTER_PRJ_C_FY2025.csv"
    db_path = base_dir / "data" / "article_grant_db.sqlite"

    print("CSV path:", csv_path)
    print("DB path:", db_path)

    engine = create_engine(f"sqlite:///{db_path}")

    grants_df = pd.read_csv(csv_path)

    grants_df = grants_df[[
        "APPLICATION_ID",
        "PROJECT_START",
        "ACTIVITY",
        "TOTAL_COST"
    ]].copy()

    #renaming the col names to match
    grants_df = grants_df.rename(columns={
        "APPLICATION_ID": "application_id",
        "PROJECT_START": "start_at",
        "ACTIVITY": "grant_type",
        "TOTAL_COST": "total_cost"
    })

    grants_df.to_sql("grants", con=engine, if_exists="append", index=False)

    print("Grants inserted successfully.")


def insert_articles(engine, data_dir):
    xml_path = data_dir / "pubmed26n1335.xml.gz"

    # parsing articles using the Articles class that was crated in articles.py
    articles = Articles(str(xml_path))
    articles_df = articles.get_entries()

    #using only the col names we need for the SQL table
    articles_df = articles_df[["PMID", "ArticleTitle"]].copy()

    articles_df = articles_df.rename(columns={
        "PMID": "pmid",
        "ArticleTitle": "title"
    })

    articles_df.to_sql("articles", con=engine, if_exists="append", index=False)
    print("Articles inserted successfully.")

#this will insert article grant relationships into the brigde table we have in the database
def insert_article_grant(engine, data_dir):
    xml_path = data_dir / "pubmed26n1335.xml.gz"

    articles = Articles(str(xml_path))
    article_grants_df = articles.get_article_grants()

    if article_grants_df.empty:
        print("No article-grant relationships found in the XML file.")
        return

    article_ids = pd.read_sql("SELECT id, pmid FROM articles", con=engine)
    grant_ids = pd.read_sql("SELECT id, application_id FROM grants", con=engine)

    article_grants_df["pmid"] = article_grants_df["pmid"].astype(str)
    article_ids["pmid"] = article_ids["pmid"].astype(str)

    article_grants_df["grant_id_text"] = article_grants_df["grant_id_text"].astype(str)
    grant_ids["application_id"] = grant_ids["application_id"].astype(str)

    bridge_df = article_grants_df.merge(article_ids, on="pmid", how="inner")
    bridge_df = bridge_df.merge(
        grant_ids,
        left_on="grant_id_text",
        right_on="application_id",
        how="inner"
    )

    bridge_df = bridge_df[["id_x", "id_y"]].drop_duplicates()
    bridge_df.columns = ["article_id", "grant_id"]

    bridge_df.to_sql("article_grant", con=engine, if_exists="append", index=False)
    print("Article-grant bridge inserted successfully.")

#this is the function that will add data to the table authors in the database, so it will fill in columns like forename, surname, and ids that will link them to  articles
def insert_authors(engine, data_dir):
    xml_path = data_dir / "pubmed26n1335.xml.gz"

    articles = Articles(str(xml_path))
    authors_df = articles.get_authors()

    if authors_df.empty:
        print("No authors found in the XML file.")
        return

    article_ids = pd.read_sql("SELECT id, pmid FROM articles", con=engine)
    article_ids["pmid"] = article_ids["pmid"].astype(str)
    authors_df["PMID"] = authors_df["PMID"].astype(str)

    authors_df = authors_df.merge(article_ids, left_on="PMID", right_on="pmid", how="inner")
    authors_df = authors_df[["id", "ForeName", "LastName", "Initials", "Affiliation"]]
    authors_df.columns = ["article_id", "forename", "surname", "initials", "affiliation"]

    authors_df.to_sql("authors", con=engine, if_exists="append", index=False)
    print("Authors inserted successfully.")

# this will set up the database then insert the new data into it when we run this code
def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    db_path = data_dir / "article_grant_db.sqlite"

    engine = create_engine(f"sqlite:///{db_path}")

    insert_grants(engine, data_dir)
    insert_articles(engine, data_dir)
    insert_article_grant(engine, data_dir)
    insert_authors(engine, data_dir)

if __name__ == "__main__":
    main()