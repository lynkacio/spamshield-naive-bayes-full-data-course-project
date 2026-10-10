# Data

`raw/SpamAssassin_H2_Data_FULL.xlsx` is the full project dataset supplied for this repository.
It contains **5,921 de-duplicated email texts**: **4,109 ham** and **1,812 spam**, drawn from the
`easy_ham`, `easy_ham_2`, `hard_ham`, `spam`, and `spam_2` SpamAssassin corpus groups.

Source: Apache SpamAssassin public corpus — https://spamassassin.apache.org/old/publiccorpus/

## Loader note

The workbook contains 102 legacy non-printable XML control bytes embedded in historical email
bodies. `spamshield.data.load_xlsx_records` removes only those invalid XML 1.0 bytes **in memory**
before parsing. The raw workbook itself is kept unchanged for provenance.

The MIT license in this repository covers project code, not third-party message content.
