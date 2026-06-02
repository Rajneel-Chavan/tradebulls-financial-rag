"""
Multi-source document loader for Tradebulls financial data.

Sources:
  1. YouTube transcripts (Tradebulls market commentary)
  2. Web scraping (Tradebulls website content)
  3. PDF financial reports (daily research reports)
  4. Structured FAQs (trading platform FAQs)
"""

import os
import glob
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
)

from backend.config import (
    YOUTUBE_VIDEO_ID,
    WEB_SCRAPE_URL,
    PDF_DIRECTORY,
)


def load_youtube_transcript(video_id: str = YOUTUBE_VIDEO_ID) -> list[Document]:
    """Load and structure YouTube transcript with metadata."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        full_text = " ".join(
            snippet.text for snippet in transcript
        )

        doc = Document(
            page_content=full_text,
            metadata={
                "source": f"youtube_{video_id}",
                "source_type": "youtube",
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "char_count": len(full_text),
                "loaded_at": datetime.now().isoformat(),
            },
        )
        print(f"[YouTube] Loaded transcript: {len(full_text)} chars from {video_id}")
        return [doc]

    except Exception as e:
        print(f"[YouTube] Failed to load transcript: {e}")
        return []


def load_web_content(url: str = WEB_SCRAPE_URL) -> list[Document]:
    """Scrape web page content with metadata tagging."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "source_type": "web",
                "url": url,
                "loaded_at": datetime.now().isoformat(),
            })

        total_chars = sum(len(d.page_content) for d in docs)
        print(f"[Web] Loaded {len(docs)} page(s): {total_chars} chars from {url}")
        return docs

    except Exception as e:
        print(f"[Web] Failed to load from {url}: {e}")
        return []


def load_pdf_reports(directory: str = PDF_DIRECTORY) -> list[Document]:
    """Load all PDF reports from directory with financial metadata."""
    all_docs = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))

    if not pdf_files:
        print(f"[PDF] No PDF files found in {directory}/")
        return []

    for pdf_path in pdf_files:
        try:
            loader = UnstructuredPDFLoader(pdf_path, mode="elements")
            docs = loader.load()

            filename = os.path.basename(pdf_path)
            for doc in docs:
                doc.metadata.update({
                    "source_type": "pdf",
                    "filename": filename,
                    "file_path": pdf_path,
                    "loaded_at": datetime.now().isoformat(),
                })

            all_docs.extend(docs)
            print(f"[PDF] Loaded {filename}: {len(docs)} elements")

        except Exception as e:
            print(f"[PDF] Failed to load {pdf_path}: {e}")

    print(f"[PDF] Total: {len(all_docs)} elements from {len(pdf_files)} file(s)")
    return all_docs


def load_faqs(faq_text: str | None = None) -> list[Document]:
    """
    Load structured FAQs with metadata.

    Pass raw FAQ text or it will use a default Tradebulls FAQ set.
    FAQs are split on the → separator for clean boundaries.
    """
    if faq_text is None:
        faq_text = _default_tradebulls_faqs()

    faq_entries = faq_text.strip().split("→")
    docs = []

    for i, entry in enumerate(faq_entries):
        entry = entry.strip()
        if not entry:
            continue

        doc = Document(
            page_content=entry,
            metadata={
                "source": f"faq_{i + 1}",
                "source_type": "faq",
                "faq_index": i + 1,
                "loaded_at": datetime.now().isoformat(),
            },
        )
        docs.append(doc)

    print(f"[FAQ] Loaded {len(docs)} FAQ entries")
    return docs


def load_all_sources() -> list[Document]:
    """Load all 4 data sources and return unified document list."""
    all_docs = []

    print("\n" + "=" * 60)
    print("LOADING ALL TRADEBULLS DATA SOURCES")
    print("=" * 60)

    # Source 1: YouTube
    all_docs.extend(load_youtube_transcript())

    # Source 2: Web
    all_docs.extend(load_web_content())

    # Source 3: PDFs
    all_docs.extend(load_pdf_reports())

    # Source 4: FAQs
    all_docs.extend(load_faqs())

    print(f"\n{'=' * 60}")
    print(f"TOTAL DOCUMENTS LOADED: {len(all_docs)}")

    # Print source breakdown
    source_counts = {}
    for doc in all_docs:
        src = doc.metadata.get("source_type", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    for src, count in source_counts.items():
        print(f"  {src}: {count}")
    print("=" * 60 + "\n")

    return all_docs


def _default_tradebulls_faqs() -> str:
    """Default Tradebulls trading platform FAQs."""
    return """
What is Tradebulls Securities?
Tradebulls Securities (P) Ltd is a SEBI-registered stock broking firm offering equity,
commodity, and currency trading services across NSE, BSE, and MCX exchanges.
→
How do I open a demat account with Tradebulls?
You can open a demat account online through the Tradebulls website or app.
Required documents include PAN card, Aadhaar card, bank statement, and a passport-size photo.
The account opening process is paperless and typically completed within 24 hours.
→
What trading platforms does Tradebulls offer?
Tradebulls offers multiple trading platforms including a web-based trading terminal,
a mobile trading app for Android and iOS, and desktop trading software.
All platforms provide real-time market data, charts, and order placement capabilities.
→
What are the brokerage charges at Tradebulls?
Tradebulls offers competitive brokerage plans. Equity delivery trades start from 0.10%
of trade value. Intraday and F&O trades are available at flat rates.
Detailed brokerage plans are available on the Tradebulls website.
→
How can I access Tradebulls research reports?
Daily research reports are published every trading day before market hours.
Reports cover Nifty outlook, sector analysis, stock recommendations with entry/exit levels,
and derivative strategies. Reports are accessible via the client portal and email.
→
What is the customer support process at Tradebulls?
Tradebulls provides customer support via phone, email, and live chat during market hours.
Support is available for account queries, technical issues, and trading assistance.
The support team operates from 9 AM to 6 PM on all trading days.
→
What types of orders can I place on Tradebulls?
Tradebulls supports market orders, limit orders, stop-loss orders, bracket orders,
cover orders, and after-market orders (AMO). All order types are available across
equity, F&O, commodity, and currency segments.
→
Does Tradebulls provide margin trading?
Yes, Tradebulls provides margin trading facilities for intraday trading.
Margin exposure varies by segment and stock category. Margin requirements
are updated daily based on exchange guidelines and volatility.
"""
