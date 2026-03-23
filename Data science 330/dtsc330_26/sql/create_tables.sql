CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pmid VARCHAR(20) NOT NULL,
  title TEXT
);

CREATE TABLE IF NOT EXISTS grants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  application_id VARCHAR(20) NOT NULL,
  start_at DATE,
  grant_type VARCHAR(10),
  total_cost INTEGER

);

CREATE TABLE IF NOT EXISTS article_grant (
  article_id INTEGER,
  grant_id INTEGER,
  FOREIGN KEY (article_id) REFERENCES articles(id),
  FOREIGN KEY (grant_id) REFERENCES grants(id)
);

CREATE TABLE IF NOT EXISTS authors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id INTEGER,
  forename VARCHAR(50),
  lastname VARCHAR(50),
  FOREIGN KEY (article_id) REFERENCES articles(id)
);

