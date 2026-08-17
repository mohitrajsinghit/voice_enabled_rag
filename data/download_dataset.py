"""Download a subset of ai4bharat/MSMARCO-XI from HuggingFace and save for RAG processing."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


SAMPLE_DATA = [
    {
        "query_id": "s0",
        "Eng_Query": "What is the Taj Mahal and where is it located?",
        "query": "ताजमहल क्या है और यह कहाँ स्थित है?",
        "Eng_Answer": "The Taj Mahal is an ivory-white marble mausoleum located on the south bank of the Yamuna river in Agra, Uttar Pradesh, India.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "English_passages": [
                "The Taj Mahal is an ivory-white marble mausoleum located on the south bank of the Yamuna river in Agra, Uttar Pradesh, India. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal.",
                "Agra is a historic city on the banks of the Yamuna river in Uttar Pradesh, India. It is famous worldwide for the Taj Mahal, Agra Fort, and Fatehpur Sikri.",
                "Shah Jahan was the fifth Mughal emperor of India from 1628 to 1658. He erected many monuments, the most famous of which is the Taj Mahal located in Agra, India.",
            ],
            "Translated_passages": [
                "ताजमहल भारत के उत्तर प्रदेश के आगरा में यमुना नदी के दक्षिणी तट पर स्थित एक हाथीदांत-सफेद संगमरमर का मकबरा है।",
                "आगरा उत्तर प्रदेश, भारत में यमुना नदी के तट पर एक ऐतिहासिक शहर है।",
                "शाहजहाँ 1628 से 1658 तक भारत का पाँचवाँ मुग़ल सम्राट था।",
            ],
            "is_selected": [1, 0, 0],
        },
    },
    {
        "query_id": "s1",
        "Eng_Query": "Who built the Red Fort in Delhi?",
        "query": "दिल्ली में लाल किला किसने बनवाया था?",
        "Eng_Answer": "The Red Fort was built by the Mughal emperor Shah Jahan in 1639.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "English_passages": [
                "The Red Fort is a historic fort in Old Delhi, Delhi in India. It was built by Mughal emperor Shah Jahan in 1639 when he decided to shift his capital from Agra to Delhi.",
                "Delhi is the capital city of India. It has been continuously inhabited since the 6th century BCE and through most of its history, Delhi has served as a capital of various kingdoms.",
            ],
            "Translated_passages": [
                "लाल किला भारत में पुरानी दिल्ली, दिल्ली में एक ऐतिहासिक किला है। इसे मुग़ल बादशाह शाहजहाँ ने 1639 में बनवाया था।",
                "दिल्ली भारत की राजधानी है।",
            ],
            "is_selected": [1, 0],
        },
    },
    {
        "query_id": "s2",
        "Eng_Query": "What is Python programming language used for?",
        "query": "पायथन प्रोग्रामिंग भाषा का उपयोग किस लिए किया जाता है?",
        "Eng_Answer": "Python is used for web development, data science, artificial intelligence, automation, and general software development.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "English_passages": [
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. Python is dynamically-typed and garbage-collected.",
                "Python is commonly used for developing websites and software, task automation, data analysis, and data visualization. Because it's relatively easy to learn, Python has been adopted by many non-programmers.",
            ],
            "Translated_passages": [
                "पायथन एक उच्च-स्तरीय, सामान्य-उद्देश्यीय प्रोग्रामिंग भाषा है।",
                "पायथन का उपयोग आमतौर पर वेबसाइटों और सॉफ्टवेयर विकसित करने, डेटा विश्लेषण और डेटा विज़ुअलाइज़ेशन के लिए किया जाता है।",
            ],
            "is_selected": [0, 1],
        },
    },
    {
        "query_id": "s3",
        "Eng_Query": "What is a sliding window algorithm or protocol?",
        "query": "स्लाइडिंग विंडो क्या है?",
        "Eng_Answer": "A sliding window is a computational technique or network protocol where a fixed or variable-size window moves across a sequential dataset or stream to maintain rolling metrics or control packet transmission.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "English_passages": [
                "The sliding window technique is an algorithmic method used to perform required operations on a specific window size of a given array or data stream, such as finding subarray sums or substrings.",
                "In computer networking, sliding window is a flow control protocol used in TCP to ensure reliable, in-order packet delivery between sender and receiver.",
            ],
            "Translated_passages": [
                "स्लाइडिंग विंडो तकनीक एक एल्गोरिथम विधि है जिसका उपयोग डेटा स्ट्रीम या सरणी पर संचालन करने के लिए किया जाता है।",
                "कंप्यूटर नेटवर्किंग में, स्लाइडिंग विंडो टीसीपी में उपयोग किया जाने वाला एक फ्लो कंट्रोल प्रोटोकॉल है।",
            ],
            "is_selected": [1, 0],
        },
    },
    {
        "query_id": "s4",
        "Eng_Query": "Why do humans sleep and what is sleep hygiene?",
        "query": "इंसान क्यों सोते हैं?",
        "Eng_Answer": "Humans sleep for brain restoration, memory consolidation, tissue repair, and immune system health.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "English_passages": [
                "Sleep is an essential biological process for humans. During sleep, the brain clears metabolic waste, consolidates memories, and repairs neural circuits.",
                "Good sleep hygiene includes keeping a consistent sleep schedule, avoiding blue light before bed, maintaining a dark cool bedroom, and avoiding caffeine late in the day.",
            ],
            "Translated_passages": [
                "नींद इंसानों के लिए एक आवश्यक जैविक प्रक्रिया है।",
                "अच्छी नींद के लिए नियमित दिनचर्या और शांत वातावरण आवश्यक है।",
            ],
            "is_selected": [1, 0],
        },
    },
]


def download_dataset(n_docs: int = 1100, output_dir: str | None = None) -> None:
    """Download MSMARCO-XI dataset and extract passages + queries.

    Args:
        n_docs: Number of query-passage groups to download.
        output_dir: Output directory for raw data files.
    """
    raw_dir = Path(output_dir) if output_dir else Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    passages_path = raw_dir / "passages.jsonl"
    queries_path = raw_dir / "queries.jsonl"

    logging.info(f"Fetching MSMARCO-XI dataset subset (up to {n_docs} groups)...")

    dataset_iterable = []
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        logging.info("Downloading MSMARCO-XI validation subset from HuggingFace...")
        parquet_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename="validation/hinval.parquet",
            repo_type="dataset",
        )
        pf = pq.ParquetFile(parquet_path)
        rows_read = 0
        df_list = []
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            df_batch = table.to_pandas()
            df_list.append(df_batch)
            rows_read += len(df_batch)
            if rows_read >= n_docs:
                break
        full_df = pd.concat(df_list, ignore_index=True).iloc[:n_docs]
        dataset_iterable = SAMPLE_DATA + full_df.to_dict(orient="records")
        logging.info(f"Successfully extracted {len(dataset_iterable)} records from HuggingFace (+ sample items).")
    except Exception as e:
        logging.warning(f"Could not load from HuggingFace ({e}). Using sample dataset.")
        dataset_iterable = SAMPLE_DATA

    passage_count = 0
    query_count = 0

    with open(passages_path, "w", encoding="utf-8") as pf, \
         open(queries_path, "w", encoding="utf-8") as qf:

        for idx, example in enumerate(dataset_iterable):
            query_id = str(example.get("query_id", idx))
            eng_query = example.get("Eng_Query", "") or example.get("query", "")
            translated_query = example.get("query", "")
            eng_answer = example.get("Eng_Answer", "") or example.get("Answer", "")
            source_lang = example.get("source_lang", "eng_Latn")
            target_lang = example.get("target_lang", "hin_Deva")

            passages = example.get("passages", {})
            eng_passages = passages.get("English_passages", []) if isinstance(passages, dict) else []
            translated_passages = passages.get("Translated_passages", []) if isinstance(passages, dict) else []
            is_selected = passages.get("is_selected", []) if isinstance(passages, dict) else []

            relevant_doc_ids = []

            for p_idx, eng_text in enumerate(eng_passages):
                if not eng_text or not eng_text.strip():
                    continue

                doc_id = f"doc_{query_id}_{p_idx}"
                trans_text = str(translated_passages[p_idx]) if p_idx < len(translated_passages) else ""
                selected = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0

                passage_record = {
                    "doc_id": str(doc_id),
                    "text": str(eng_text).strip(),
                    "translated_text": trans_text.strip() if trans_text else "",
                    "source_lang": str(source_lang),
                    "target_lang": str(target_lang),
                    "query_id": str(query_id),
                    "is_selected": int(selected),
                }
                pf.write(json.dumps(passage_record, ensure_ascii=False, default=str) + "\n")
                passage_count += 1

                if selected == 1:
                    relevant_doc_ids.append(str(doc_id))

            query_record = {
                "query_id": str(query_id),
                "query": str(eng_query).strip() if eng_query else str(translated_query).strip(),
                "translated_query": str(translated_query).strip(),
                "answer": str(eng_answer).strip() if eng_answer else "",
                "source_lang": str(source_lang),
                "target_lang": str(target_lang),
                "relevant_doc_ids": relevant_doc_ids,
            }
            qf.write(json.dumps(query_record, ensure_ascii=False, default=str) + "\n")
            query_count += 1

    logging.info(f"Saved {passage_count} passages to {passages_path}")
    logging.info(f"Saved {query_count} queries to {queries_path}")
    logging.info("Dataset preparation complete!")


def main():
    parser = argparse.ArgumentParser(description="Download MSMARCO-XI dataset subset")
    parser.add_argument("--n-docs", type=int, default=5000, help="Number of query-passage groups to download")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    download_dataset(n_docs=args.n_docs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
