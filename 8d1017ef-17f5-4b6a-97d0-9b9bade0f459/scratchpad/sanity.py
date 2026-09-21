from tools import document_fields as df

print('kvitancia root match against "No kvitancii":', bool(df._keyword_regex("квитанция").search("№ квитанции")))

m = df._LABEL_VALUE_RE.match("31.08.2026 21:48")
print("label-value match on date+time line (should be None):", m)

m2 = df._LABEL_VALUE_RE.match("VIN: 1HGCM82633A004352")
print("label-value match on VIN line (should work):", m2.groups() if m2 else None)
