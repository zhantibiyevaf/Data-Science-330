import gzip
import xml.etree.ElementTree as ET

import pandas as pd


class Articles:
    def __init__(self, path: str):
        """Read in a pubmed articles XML file that is gzipped

        Args:
            path (str): location of GZIPped file on the disk
        """
        self.article_df = None
        self.author_df = None
        self._parse(path)  # I'm being inconsistent-- don't be

    def _parse(self, path: str):
        """Parse the Pubmed file"""
        articles = []
        authors = []
        # One trick for creating a dataframe is to create a list of
        # dicts with the same naming format
        with gzip.open(path, "rb") as fp:
            # _ means throw it away
            # for test in ET.iterparse(fp, events=('end',)):
            # test = (index, value itself)

            for _, article in ET.iterparse(fp, events=("end",)):
                if article.tag == "PubmedArticle":
                    article_row, article_authors = self._parse_article(article)
                    articles.append(article_row)
                    authors.extend(article_authors)  # be careful extend vs append
                    # append: [[auth1, auth2], [auth3, auth4, auth5]]
                    # extend: [auth1, auth2, ..., auth5]
                    article.clear()

        self.article_df = pd.DataFrame(articles)
        self.author_df = pd.DataFrame(authors)

    def _parse_article(self, article: ET.Element):
        """Parse an XML PubmedArticle element"""
        row = {}
        tags = ["PMID", "ArticleTitle", "Affiliation"]
        for el in article.iter():
            if el.tag in tags:
                row[el.tag] = el.text

        # Keep date parsing explicit to avoid collisions between different date tags.
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

        if "PMID" not in row.keys():
            return {}, {}

        # In XML, strictly speaking, there's no rule about order
        # <AuthorList></AuthorList><PMID></PMID>
        # <PMID></PMID><AuthorList></AuthorList>
        authors = []
        tags = ["LastName", "ForeName", "Initials", "Affiliation"]
        for author in article.findall(".//Author"):
            auth_row = {"PMID": row["PMID"]}
            for el in author.iter():
                if el.tag in tags:
                    auth_row[el.tag] = el.text
            authors.append(auth_row)

        return row, authors

    def get_authors(self):
        """Get parsed grants"""
        return self.author_df

    def get_entries(self):
        """Get parsed articles"""
        return self.article_df


if __name__ == "__main__":
    articles = Articles("data/pubmed25n1275.xml.gz")
