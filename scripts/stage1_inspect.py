import fitz

doc = fitz.open("Science_1_SSC_Testbook.pdf")
print(f"Total pages: {len(doc)}")
print()
for i in range(0, 45):
    page = doc[i]
    text = page.get_text("text").strip()
    first_lines = " | ".join(l.strip() for l in text.splitlines()[:5] if l.strip())
    print(f"Page {i+1:3d}: {first_lines[:130]}")
doc.close()
