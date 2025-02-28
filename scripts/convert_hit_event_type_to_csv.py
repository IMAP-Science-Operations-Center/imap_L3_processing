import docx

filename = "tests/test_data/hit/pha_events/HIT_Event_Types_250211.docx"
doc = docx.Document(filename)
fullText = []
table = doc.tables[0]
with open("imap_hit_l3_hit-event-types-text-not-cdf_v001.cdf", "w") as file:
    for row in table.rows:
        file.write(",".join([cell.text for cell in row.cells]) + "\n")
