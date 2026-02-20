from dotenv import load_dotenv
load_dotenv()

from ingestion import split_documents, load_pdfs, save_documents_to_db

pdf_files = [
    "./resources/gida_hijyeni_yonetmeligi.pdf",
    "./resources/toplu_tuketim_yerleri_hijyen_uygulama_kilavuzu.pdf"
    # "./resources/isyeri_acma_ve_calisma_ruhsatlarina_iliskin_yonetmelik.pdf",
]

all_documents=load_pdfs(pdf_files)

documents=split_documents(all_documents)

save_documents_to_db(documents)