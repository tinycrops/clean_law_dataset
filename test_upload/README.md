---
license: mit
datasets:
- the-ride-never-ends/american_law
language:
- en
size_categories:
- n>10K
---

# American Law Citations Dataset

This dataset contains 687 legal citations extracted from the American Law dataset.

## Dataset Description

The dataset contains parsed legal citations from various legal documents, including:

- U.S. Code (USC): 219 citations
- Code of Federal Regulations (CFR): 301 citations
- Public Laws: 124 citations
- Statutes at Large: 43 citations

Each citation includes the original document ID, document title, citation type, the full citation text, 
context surrounding the citation, and a SQL query that can be used to retrieve the cited document from a relational database.

## Dataset Creation

This dataset was created by parsing the text from the American Law dataset and extracting structured information 
about legal citations using regular expressions. The processing script identifies different types of legal citations 
and extracts relevant information such as title, section, and part numbers.

## Citation Types

1. **USC (United States Code)**: References to federal statutory law, e.g., "17 U.S.C. 501"
2. **CFR (Code of Federal Regulations)**: References to federal regulations, e.g., "40 CFR 261"
3. **Public Law**: References to laws as originally passed, e.g., "Public Law 89-655"
4. **Stat (Statutes at Large)**: References to the official publication of laws, e.g., "124 Stat. 119"

## Sample SQL Queries

The dataset includes SQL queries that can be used to retrieve the cited documents from a relational database:

- USC: `SELECT * FROM usc WHERE title = '17' AND section = '501'`
- CFR: `SELECT * FROM cfr WHERE title = '40' AND part = '261'`
- Public Law: `SELECT * FROM public_laws WHERE congress = '89' AND law_number = '655'`
- Stat: `SELECT * FROM statutes WHERE volume = '124' AND page = '119'`

## Usage

This dataset can be used for legal information retrieval, citation analysis, and building systems that link to primary legal sources.
