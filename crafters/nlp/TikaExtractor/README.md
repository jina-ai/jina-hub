# TikaExtractor

Based on Apache Tika, extracts text "from over a thousand different file types (such as PPT, XLS, and PDF)".

It accepts file URI and it will return text of that file detected by Apache Tika.

Following environment configuration is supported:

```bash
TIKA_OCR_STRATEGY (default: ocr_only)
TIKA_EXTRACT_INLINE_IMAGES (default: true)
TIKA_OCR_LANGUAGE (default: eng)
TIKA_TIMEOUT (default: 600)
```
