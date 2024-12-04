from pdf2docx import Converter

pdf_file = "Bai 2.pdf"
docx_file = "cloding2.docx"

cv = Converter(pdf_file)
cv.convert(docx_file)
cv.close()