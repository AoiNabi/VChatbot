import fitz

class PDFReader:
    def __init__(self):
        print("📄 Lector de PDF inicializado.")

    def extract_text(self, pdf_path):
        """
        Extrae el texto de un archivo PDF completo.

        Args:
            pdf_path (str): Ruta del archivo PDF.

        Returns:
            str: Texto completo del PDF.
        """
        if not pdf_path:
            return "[ERROR: Ruta de archivo vacía]"

        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page_num, page in enumerate(doc, start=1):
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"❌ Error al leer el PDF: {e}")
            return f"[ERROR AL LEER PDF: {e}]"
