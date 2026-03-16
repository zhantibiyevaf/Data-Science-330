import gzip
import xml.etree.ElementTree as ET

import pandas as pd


class Articles:
    def __init__(self, path: str):
        """Read in a pubmed articles XML file that is gzipped."""
        self.article_df = None
        self.author_df = None
        self.article_grant_df = None
        self._parse(path)

    def _parse(self, path: str):
        """Parse the Pubmed file."""
        articles = []
        authors = []
        article_grants = []

        with gzip.open(path, "rb") as fp:
            for _, article in ET.iterparse(fp, events=("end",)):
                if article.tag == "PubmedArticle":
                    article_row, article_authors, article_grant_rows = self._parse_article(article)

                    if article_row:
                        articles.append(article_row)

                    authors.extend(article_authors)
                    article_grants.extend(article_grant_rows)

                    article.clear()

        self.article_df = pd.DataFrame(articles)
        self.author_df = pd.DataFrame(authors)
        self.article_grant_df = pd.DataFrame(article_grants)

    def _parse_article(self, article: ET.Element):
        """Parse an XML PubmedArticle element."""
        row = {}
        tags = ["PMID", "ArticleTitle", "Affiliation"]
        for el in article.iter():
            if el.tag in tags:
                row[el.tag] = el.text

        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            for part in ("Year", "Month", "Day"):
                part_el = pub_date.find(part)
                if part_el is not None:
                    row[f"pub_{part.lower()}"] = part_el.text

        completed = article.find(".//DateCompleted")
        if completed is not None:
            for part in ("Year", "Month", "Day"):
                part_el = completed.find(part)
                if part_el is not None:
                    row[f"completed_{part.lower()}"] = part_el.text

        if "PMID" not in row:
            return {}, [], []

        authors = []
        author_tags = ["LastName", "ForeName", "Initials", "Affiliation"]
        for author in article.findall(".//Author"):
            auth_row = {"PMID": row["PMID"]}
            for el in author.iter():
                if el.tag in author_tags:
                    auth_row[el.tag] = el.text
            authors.append(auth_row)

        article_grants = []
        for grant in article.findall(".//Grant"):
            grant_id = grant.findtext("GrantID")
            acronym = grant.findtext("Acronym")
            agency = grant.findtext("Agency")
            country = grant.findtext("Country")

            if grant_id:
                article_grants.append(
                    {
                        "pmid": row["PMID"],
                        "grant_id_text": grant_id,
                        "acronym": acronym,
                        "agency": agency,
                        "country": country,
                    }
                )

        return row, authors, article_grants

    def get_authors(self):
        return self.author_df

    def get_entries(self):
        return self.article_df

    def get_article_grants(self):
        return self.article_grant_df


if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[2] / "data"
    articles = Articles(str(data_dir / "pubmed26n1335.xml.gz"))
    print(articles.get_entries().head())
    print(articles.get_authors().head())
    print(articles.get_article_grants().head())