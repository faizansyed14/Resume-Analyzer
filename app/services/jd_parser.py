# app/services/jd_parser.py
import os
import re
import io
from typing import Dict, Optional, Tuple, List

# ---- File & OCR libs ----
import fitz  # PyMuPDF
import pdfplumber
import docx
import pytesseract
from PIL import Image

# ---- NLP libs ----
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

# ---- Optional NER/embeddings ----
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

from .skill_library import SkillLibrary


class JDParser:
    """
    Parser output:
    {
      "job_title": str,
      "company": str,
      "location": str,
      "skills": [str, ...],
      "experience_requirements": [str, ...],
      "education_requirements": [str, ...],
      "min_years_experience": int | None,
      "certifications_required": [str, ...],
      "certifications_preferred": [str, ...],
      "languages": { "programming": [..], "spoken": [..] },
      "summary_objective": str
    }
    """

    OCR_DPI = 300
    OCR_PSM = 6
    OCR_OEM = 3

    def __init__(self):
        self.nlp = self._load_spacy_pipeline()

        self.ner_pipeline = None
        if HF_AVAILABLE:
            try:
                tok = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                mdl = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                self.ner_pipeline = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
                print("HF NER pipeline loaded for JD parsing.")
            except Exception as e:
                print(f"HF NER not available: {e}")

        if ST_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
                print("SentenceTransformer ready (optional).")
            except Exception as e:
                print(f"SentenceTransformer load failed: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None

        # Skills
        self.skill_library = SkillLibrary()
        self.all_skills: List[str] = self.skill_library.get_all_skills()
        self.skills_lower_map: Dict[str, str] = getattr(self.skill_library, "skills_lowercase", {
            s.lower(): s for s in self.all_skills
        })

        self.skill_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.skill_matcher.add("SKILLS", [self.nlp.make_doc(s) for s in self.all_skills])

        self.headers = {
            "title": ["job title", "position", "role", "title"],
            "summary": ["job description", "overview", "summary", "about the role", "about us"],
            "skills": ["skills", "key skills", "skills / strengths / key skills", "required skills",
                       "technical skills", "what you’ll need", "what we're looking for", "qualifications"],
            "experience": ["experience", "experience / responsibilities / requirements", "responsibilities",
                           "requirements", "duties", "what you’ll do", "role responsibilities"],
            "education": ["qualification", "qualification / education", "education", "education requirements",
                          "academic qualifications"],
            "certifications": ["certifications", "licenses", "industry certifications", "credentials"],
            "location": ["location", "based in"],
            "company": ["company", "about the company", "about us"]
        }

        # Regex
        self._re_spaces = re.compile(r"[ \t]+")
        self._re_newlines = re.compile(r"\n{3,}")
        self._re_cr = re.compile(r"\r")
        self._re_bullet_line = re.compile(r"^\s*(?:[-*•▪▫◦‣⁃]\s+|\d+\.\s+)")
        self._re_header_any = re.compile(r"(?m)^\s*[A-Z][A-Za-z0-9 &/]{2,60}\s*$")

        self._re_years_single = re.compile(r"(?i)(\d{1,2})\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp)\b")
        self._re_years_range = re.compile(r"(?i)(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*years?\s+(?:of\s+)?(?:experience|exp)\b")

        self._re_required_cert = re.compile(r"(?i)\b(?:mandatory|required|must\s*have)\b\s*:\s*(.+)")
        self._re_preferred_cert = re.compile(r"(?i)\b(?:preferred|nice\s*to\s*have|good\s*to\s*have|plus|an\s*advantage)\b\s*:?\s*(.+)?")

        self._spoken_langs = set([l.lower() for l in self.skill_library.get_skills_by_category("spoken_languages")])

        print("JDParser ready.")

    # ---------- Public API ----------
    def parse_job_pdf(self, pdf_path: str) -> Tuple[str, Dict, Optional[Dict]]:
        ext = os.path.splitext(pdf_path)[1].lower()
        if ext != ".pdf":
            raise ValueError(f"Expected a PDF; got {ext}")

        text = self._extract_pdf_text(pdf_path)
        if not text.strip():
            print("No extractable text; falling back to OCR.")
            text = self._ocr_pdf(pdf_path)

        cleaned = self._clean_text(text)
        entities = self._extract_entities(cleaned)
        return text, entities, None

    def parse_job_docx(self, docx_path: str) -> Tuple[str, Dict, Optional[Dict]]:
        text = self._extract_docx_text(docx_path)
        cleaned = self._clean_text(text)
        entities = self._extract_entities(cleaned)
        return text, entities, None

    def parse_job_text(self, text: str) -> Tuple[str, Dict, Optional[Dict]]:
        cleaned = self._clean_text(text or "")
        entities = self._extract_entities(cleaned)
        return text, entities, None

    # ---------- Extraction core ----------
    def _extract_entities(self, text: str) -> Dict:
        title = self._extract_job_title(text)
        company = self._extract_company(text)
        location = self._extract_location(text)

        summary = self._extract_section(text, self.headers["summary"])
        skills_block = self._extract_section(text, self.headers["skills"])
        exp_block = self._extract_section(text, self.headers["experience"])
        edu_block = self._extract_section(text, self.headers["education"])
        cert_block = self._extract_section(text, self.headers["certifications"])

        skills = self._extract_skills(skills_block or text)

        experience_requirements = self._bullets_or_sentences(exp_block)
        education_requirements = self._bullets_or_sentences(edu_block)

        min_years = self._parse_min_years_experience(" ".join([exp_block or "", edu_block or "", summary or "", text[:2000]]))

        cert_required, cert_preferred = self._parse_certifications(cert_block, edu_block)

        languages = self._extract_languages(skills, text)

        return {
            "job_title": title,
            "company": company,
            "location": location,
            "skills": skills,
            "experience_requirements": experience_requirements,
            "education_requirements": education_requirements,
            "min_years_experience": min_years,
            "certifications_required": cert_required,
            "certifications_preferred": cert_preferred,
            "languages": languages,
            "summary_objective": summary.strip() if summary else ""
        }

    # ---------- Sections ----------
    def _extract_section(self, text: str, headers: List[str]) -> str:
        if not text or not headers:
            return ""
        header_pattern = r"(?im)^\s*(?:" + "|".join(re.escape(h) for h in headers) + r")\s*[:\-–]?\s*$"
        m = re.search(header_pattern, text)
        if not m:
            numbered = r"(?im)^\s*\d+\.\s*(?:" + "|".join(re.escape(h) for h in headers) + r")\s*$"
            m = re.search(numbered, text)
            if not m:
                loose = r"(?im)^\s*(?:(?:\d+\.\s*)?)(" + "|".join([re.escape(h) for h in headers]) + r")(?:\s*/.*)?\s*$"
                m = re.search(loose, text)
                if not m:
                    return ""
        start = m.end()
        next_header = self._re_header_any.search(text[start:])
        end = start + next_header.start() if next_header else len(text)
        return text[start:end].strip()

    # ---------- Title / Company / Location ----------
    def _extract_job_title(self, text: str) -> str:
        first_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if first_lines:
            m = re.match(r"(?i)^(?:job\s*title|position|role)\s*:\s*(.+)$", first_lines[0])
            if m:
                return m.group(1).strip()
            m2 = re.match(r"^([A-Za-z][A-Za-z0-9 /&\-\(\)]+):\s*$", first_lines[0])
            if m2:
                return m2.group(1).strip()
            if self._looks_like_role_line(first_lines[0]):
                return first_lines[0]
        from_header = self._extract_section(text, self.headers["title"])
        if from_header:
            return from_header.split("\n")[0].strip()
        try:
            doc = self.nlp(" ".join(first_lines[:3])[:200])
            cand = []
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text.split()) <= 6:
                    cand.append(chunk.text)
            if cand:
                return cand[0]
        except Exception:
            pass
        return first_lines[0] if first_lines else ""

    def _looks_like_role_line(self, s: str) -> bool:
        s = s.strip()
        if not (2 <= len(s.split()) <= 8):
            return False
        kw = ["engineer", "developer", "analyst", "manager", "architect",
              "specialist", "coordinator", "lead", "senior", "junior", "consultant",
              "scientist", "administrator", "designer", "director"]
        return any(k in s.lower() for k in kw)

    def _extract_company(self, text: str) -> str:
        # Prefer explicit section
        h = self._extract_section(text, self.headers["company"])
        if h:
            first = h.split("\n")[0].strip()
            m = re.match(r"(?i)^company\s*:\s*(.+)$", first)
            return (m.group(1).strip() if m else first)

        # NER (ORG) near top — but skip strings that are known skills (e.g., "MySQL")
        try:
            doc = self.nlp(text[:600])
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org = ent.text.strip()
                    if org.lower() in self.skills_lower_map:  # skip tech terms
                        continue
                    return org
        except Exception:
            pass
        return ""

    def _extract_location(self, text: str) -> str:
        h = self._extract_section(text, self.headers["location"])
        if h:
            first = h.split("\n")[0].strip()
            m = re.match(r"(?i)^location\s*:\s*(.+)$", first)
            return (m.group(1).strip() if m else first)
        try:
            doc = self.nlp(text[:600])
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    return ent.text.strip()
        except Exception:
            pass
        return ""

    # ---------- Skills ----------
    def _extract_skills(self, text: str) -> List[str]:
        if not text:
            return []
        doc: Doc = self.nlp(text)
        matches = self.skill_matcher(doc)
        found_lower = set()
        for _, start, end in matches:
            token_text = doc[start:end].text
            if self.skill_library.is_ambiguous(token_text):
                continue
            found_lower.add(token_text.lower())

        canonical: set[str] = set()
        for low in found_lower:
            if low in self.skills_lower_map:
                canonical.add(self.skills_lower_map[low])
            else:
                alias = self.skill_library.normalize_alias(low)
                if alias and alias.lower() in self.skills_lower_map:
                    canonical.add(self.skills_lower_map[alias.lower()])

        return sorted(canonical, key=lambda x: x.lower())

    # ---------- Bullets/Sentences ----------
    def _bullets_or_sentences(self, block: str | None) -> List[str]:
        if not block:
            return []
        bullets = []
        for ln in block.split("\n"):
            if self._re_bullet_line.match(ln):
                item = re.sub(self._re_bullet_line, "", ln).strip(" -•▪▫◦‣⁃—\t")
                item = self._collapse_ws(item)
                if len(item) > 1:
                    bullets.append(item)
        if bullets:
            return bullets
        sents = re.split(r"(?<=[.!?])\s+", self._collapse_ws(block))
        sents = [s.strip() for s in sents if len(s.strip()) > 1]
        return sents

    # ---------- Years of experience ----------
    def _parse_min_years_experience(self, text: str) -> Optional[int]:
        rng = self._re_years_range.search(text)
        if rng:
            try:
                lo = int(rng.group(1))
                hi = int(rng.group(2))
                return min(lo, hi)
            except Exception:
                pass
        best = None
        for m in self._re_years_single.finditer(text):
            try:
                val = int(m.group(1))
                if best is None or val < best:
                    best = val
            except Exception:
                continue
        return best

    # ---------- Certifications ----------
    def _parse_certifications(self, cert_block: str | None, edu_block: str | None) -> Tuple[List[str], List[str]]:
        required, preferred = [], []

        def harvest(lines: List[str], into: List[str]):
            for ln in lines:
                parts = re.split(r"[;,/]| and ", ln, flags=re.I)
                for p in parts:
                    s = p.strip(" :-•\t()")
                    if not s:
                        continue
                    # strip trailing noise
                    s = re.sub(r"(?i)\betc\.?\)?$", "", s).strip()
                    s = re.sub(r"(?i)\b(are|is)\s+a\s+plus\.?$", "", s).strip()
                    if len(s) >= 2 and re.search(r"[A-Za-z0-9]", s):
                        into.append(s)

        def handle_block(block: str, default_bucket: List[str]):
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for ln in lines:
                m_req = self._re_required_cert.search(ln)
                m_pref = self._re_preferred_cert.search(ln)
                if m_req and m_req.group(1):
                    harvest([m_req.group(1)], required)
                elif m_pref:
                    captured = m_pref.group(1) if m_pref.group(1) else ln
                    harvest([captured], preferred)
                else:
                    harvest([ln], default_bucket)

        if cert_block:
            handle_block(cert_block, required)

        if edu_block:
            for ln in [l.strip() for l in edu_block.split("\n") if l.strip()]:
                if re.search(r"(?i)\b(certification|certifications|certificate|certified)\b", ln) or \
                   re.search(r"(?i)\b(plus|preferred|nice to have|good to have|advantage)\b", ln):
                    if re.search(r"(?i)\b(preferred|nice to have|good to have|plus|advantage)\b", ln):
                        harvest([ln], preferred)
                    else:
                        harvest([ln], required)

        def clean_and_dedup(seq: List[str]) -> List[str]:
            out, seen = [], set()
            for x in seq:
                k = x.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            return out

        return clean_and_dedup(required), clean_and_dedup(preferred)

    # ---------- Languages ----------
    def _extract_languages(self, skills: List[str], text: str) -> Dict[str, List[str]]:
        prog = [s for s in skills if self.skill_library.get_skill_category(s) == "programming_languages"]
        spoken_found = set()
        text_low = f" {text.lower()} "
        for lang in self._spoken_langs:
            if re.search(rf"(?<![a-z]){re.escape(lang)}(?![a-z])", text_low):
                spoken_found.add(self.skill_library.skills_lowercase.get(lang, lang.title()))
        return {
            "programming": sorted(set(prog), key=lambda x: x.lower()),
            "spoken": sorted(spoken_found, key=lambda x: x.lower())
        }

    # ---------- File IO ----------
    def _extract_pdf_text(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                parts = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t:
                        parts.append(t)
                if parts:
                    return "\n".join(parts)
        except Exception as e:
            print(f"pdfplumber failed: {e}")

        try:
            doc = fitz.open(pdf_path)
            text = []
            for page in doc:
                text.append(page.get_text("text"))
            doc.close()
            all_text = "\n".join(text)
            if all_text.strip():
                return all_text
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

        return ""

    def _ocr_pdf(self, file_path: str) -> str:
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"fitz open failed in OCR: {e}")
            return ""

        text = []
        for page in doc:
            try:
                pix = page.get_pixmap(dpi=self.OCR_DPI)
                img = Image.open(io.BytesIO(pix.tobytes())).convert("L")
                config = f'--oem {self.OEM} --psm {self.PSM}'
                page_text = pytesseract.image_to_string(img, config=config)
                if page_text:
                    text.append(page_text)
            except Exception:
                continue
        doc.close()
        return "\n".join(text)

    def _extract_docx_text(self, docx_path: str) -> str:
        try:
            document = docx.Document(docx_path)
            return "\n".join([p.text for p in document.paragraphs if p.text])
        except Exception as e:
            print(f"DOCX read failed: {e}")
            return ""

    # ---------- Cleaning ----------
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\u2022", "•")
        text = self._re_cr.sub("\n", text)
        text = self._re_newlines.sub("\n\n", text)
        text = self._re_spaces.sub(" ", text)
        return text.strip()

    def _collapse_ws(self, s: str) -> str:
        return self._re_spaces.sub(" ", s).strip()

    def _load_spacy_pipeline(self):
        try:
            return spacy.load("en_core_web_trf")
        except Exception:
            pass
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli as spacy_cli
            spacy_cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
