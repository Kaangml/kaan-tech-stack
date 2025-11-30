# PDF Processing

Tools and techniques for PDF parsing, extraction, and processing.

## Tool Comparison

| Tool | Best For | OCR Support | Table Extraction | Speed |
|------|----------|-------------|------------------|-------|
| PyMuPDF (fitz) | Fast extraction, images | No (native) | Basic | Fast |
| pdfplumber | Tables, precise layout | No | Excellent | Medium |
| Docling | Complex layouts | Yes | Excellent | Medium |
| Mineru | Academic papers | Yes | Good | Slow |
| pdf2image + Tesseract | Scanned docs | Yes | No | Slow |

## PyMuPDF (fitz)

Fast, low-level PDF manipulation.

### Basic Text Extraction

```python
import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# With layout preservation
def extract_with_layout(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("blocks")  # Preserves block structure
    return text
```

### Image Extraction

```python
def extract_images(pdf_path: str, output_dir: str):
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            with open(f"{output_dir}/page{page_num}_img{img_idx}.{image_ext}", "wb") as f:
                f.write(image_bytes)
```

### PDF Manipulation

```python
# Merge PDFs
def merge_pdfs(pdf_list: list, output_path: str):
    result = fitz.open()
    for pdf in pdf_list:
        doc = fitz.open(pdf)
        result.insert_pdf(doc)
        doc.close()
    result.save(output_path)
    result.close()

# Extract specific pages
def extract_pages(pdf_path: str, pages: list, output_path: str):
    doc = fitz.open(pdf_path)
    result = fitz.open()
    for page_num in pages:
        result.insert_pdf(doc, from_page=page_num, to_page=page_num)
    result.save(output_path)

# Add watermark
def add_watermark(pdf_path: str, watermark_text: str, output_path: str):
    doc = fitz.open(pdf_path)
    for page in doc:
        rect = page.rect
        page.insert_text(
            (rect.width / 2, rect.height / 2),
            watermark_text,
            fontsize=50,
            color=(0.8, 0.8, 0.8),
            rotate=45
        )
    doc.save(output_path)
```

## pdfplumber

Best for table extraction and precise layout analysis.

### Table Extraction

```python
import pdfplumber
import pandas as pd

def extract_tables(pdf_path: str) -> list[pd.DataFrame]:
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    return tables

# With custom settings for complex tables
def extract_complex_tables(pdf_path: str) -> list:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        
        # Custom table detection settings
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        }
        
        tables = page.extract_tables(table_settings)
        return tables
```

### Layout Analysis

```python
def analyze_layout(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        
        # Get all text with positions
        chars = page.chars  # Individual characters
        words = page.extract_words()  # Grouped words
        
        # Find text in specific region
        bbox = (0, 0, 300, 100)  # x0, y0, x1, y1
        cropped = page.within_bbox(bbox)
        header_text = cropped.extract_text()
        
        return {"words": words, "header": header_text}
```

## Docling

IBM's document understanding library for complex layouts.

### Basic Usage

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to markdown
markdown = result.document.export_to_markdown()

# Export to structured format
doc_dict = result.document.export_to_dict()

# Access elements
for element in result.document.elements:
    if element.category == "table":
        print(element.to_pandas())
    elif element.category == "text":
        print(element.text)
```

### Advanced Configuration

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    table_structure_options={
        "mode": "accurate"  # or "fast"
    }
)

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert("scanned_document.pdf")
```

## Mineru (MinerU)

Specialized for academic papers and complex documents.

### Installation

```bash
pip install magic-pdf[full]
# Or with GPU
pip install magic-pdf[full-cuda]
```

### Usage

```python
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

# Initialize
reader_writer = DiskReaderWriter("./output")
pipe = UNIPipe(
    pdf_bytes=open("paper.pdf", "rb").read(),
    model_list=[],
    image_writer=reader_writer
)

# Process
pipe.pipe_classify()
pipe.pipe_parse()

# Get markdown output
md_content = pipe.pipe_mk_markdown("./output", drop_mode="none")
```

## OCR with Tesseract

For scanned documents.

```python
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf(pdf_path: str) -> str:
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300)
    
    text = ""
    for image in images:
        # OCR each page
        page_text = pytesseract.image_to_string(image, lang='eng')
        text += page_text + "\n"
    
    return text

# With Turkish support
def ocr_turkish(pdf_path: str) -> str:
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for image in images:
        # Use Turkish language pack
        page_text = pytesseract.image_to_string(image, lang='tur')
        text += page_text + "\n"
    return text
```

## RAG-Optimized Chunking

```python
from typing import List
from dataclasses import dataclass

@dataclass
class PDFChunk:
    text: str
    page_number: int
    metadata: dict

def chunk_for_rag(pdf_path: str, chunk_size: int = 1000) -> List[PDFChunk]:
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(PDFChunk(
                            text=current_chunk.strip(),
                            page_number=page_num + 1,
                            metadata={"source": pdf_path}
                        ))
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(PDFChunk(
                    text=current_chunk.strip(),
                    page_number=page_num + 1,
                    metadata={"source": pdf_path}
                ))
    
    return chunks
```

## Performance Tips

```python
# Parallel processing for multiple PDFs
from concurrent.futures import ProcessPoolExecutor

def process_pdf(pdf_path: str) -> dict:
    # Your processing logic
    return {"path": pdf_path, "pages": count_pages(pdf_path)}

def batch_process(pdf_paths: list, max_workers: int = 4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_pdf, pdf_paths))
    return results

# Memory-efficient processing for large PDFs
def stream_large_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        yield page.get_text()
        # Page is garbage collected after yield
```

## Related Resources

- [RAG Systems](../../3-ai-ml/rag-systems/README.md) - Using extracted text in RAG
- [Data Stack](../data-stack/README.md) - Processing extracted data with Pandas
- [Legal RAG Blueprint](../../99-blueprints/legal-rag-graphdb/README.md) - PDF processing in production
