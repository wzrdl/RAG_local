from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

class DocumentLoader:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type

    def load(self):
        if self.file_type == 'pdf':
            return self.load_pdf()
        elif self.file_type == 'md':
            return self.load_markdown()
        elif self.file_type == 'csv':
            return self.load_csv()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def load_pdf(self):
        loader = PyMuPDFLoader(self.file_path)
        return loader.load()

    def load_markdown(self):
        loader = UnstructuredMarkdownLoader(self.file_path)
        return loader.load()

    def load_csv(self):
        loader = CSVLoader(file_path=self.file_path)
        return loader.load()

# Example usage:
# For PDF
pdf_loader = DocumentLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf", "pdf")
pdf_pages = pdf_loader.load()

# For Markdown
md_loader = DocumentLoader("../../data_base/knowledge_db/prompt_engineering/Introduction.md", "md")
md_pages = md_loader.load()

# For CSV
csv_loader = DocumentLoader('./example_data/mlb_teams_2012.csv', 'csv')
csv_data = csv_loader.load()
