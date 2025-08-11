# app/services/llm_service.py
import os
import re
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


MONTH_MAP = {
    "jan": 1, "feb": 2, "fev": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # common alt spellings
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "sept": 9,
}


def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class LLMService:
    """
    Thin wrapper around OpenAI Chat Completions with JSON output,
    plus deterministic scoring for match calculation.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini"):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        # It's fine if there's no key for analysis-only use; only required if you call the LLM.
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model_name = model_name
        self.skill_model = SentenceTransformer("all-mpnet-base-v2")  # used for semantic skill matching

        print(f"LLMService initialized. Model: {self.model_name} | Embeddings: all-mpnet-base-v2")

    # ----------------------- Optional LLM JSON helper ------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> tuple[str, dict | None]:
        """
        Call OpenAI Chat Completions and force JSON output.
        Returns (content_str, usage_dict|None).
        """
        if not self.client:
            # No API key / not calling the LLM.
            return json.dumps({"error": "LLM disabled"}), None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            usage = resp.usage.model_dump() if resp.usage else None
            return content, usage
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return json.dumps({"error": f"LLM call failed: {e}"}), None

    # ============================ MATCH SCORING ==============================

    def calculate_match_score(
        self,
        resume_data: Dict,
        job_data: Dict,
        skill_similarity_data: Dict | None = None,
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Deterministic scoring (no LLM for numbers).
        Returns only:
          - scores
          - matched_skills
          - match_analysis

        Overall = (skills * 0.8) + (experience * 0.15) + (education * 0.05)
        All component scores in [0,1].
        """

        # 1) Collect skills (be generous: consider either *.skills or *.matched_skills)
        cand_skills = self._normalize_skills_list(
            resume_data.get("matched_skills", resume_data.get("skills", []))
        )
        job_skills = self._normalize_skills_list(
            job_data.get("matched_skills", job_data.get("skills", []))
        )

        # 2) Semantic skill overlap
        matched_skills, skills_score = self._semantic_skill_overlap(job_skills, cand_skills)

        # If caller provided external similarity data, let it inform the score slightly (but never lower it)
        if skill_similarity_data and isinstance(skill_similarity_data, dict):
            pct = float(skill_similarity_data.get("match_percentage", 0.0)) / 100.0
            skills_score = max(skills_score, _clamp01(pct))

        # 3) Experience score: parse candidate years vs job minimum years
        cand_years = self._estimate_candidate_years(resume_data)
        min_years = self._extract_min_years(job_data)
        if min_years > 0:
            experience_score = _clamp01(cand_years / min_years)
        else:
            # if job doesn't state a minimum, but candidate has any experience, give a decent baseline
            experience_score = 1.0 if cand_years > 0 else 0.0

        # 4) Education score: does resume satisfy education requirement?
        education_score = self._education_match_score(resume_data, job_data)

        # 5) Overall
        overall = (skills_score * 0.80) + (experience_score * 0.15) + (education_score * 0.05)

        # 6) Compose short analysis
        analysis = self._compose_analysis(
            skills_score=skills_score,
            experience_score=experience_score,
            education_score=education_score,
            cand_years=cand_years,
            min_years=min_years,
            matched_skills=matched_skills,
        )

        result = {
            "scores": {
                "skills": round(_clamp01(skills_score), 3),
                "experience": round(_clamp01(experience_score), 3),
                "education": round(_clamp01(education_score), 3),
                "overall": round(_clamp01(overall), 3),
            },
            "matched_skills": matched_skills,
            "match_analysis": analysis,
        }

        # We didn't call the LLM for scoring; return zeroed usage so UI shows determinate numbers.
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return result, token_usage

    # =========================== INTERVIEW Qs ================================

    def generate_interview_questions(self, context: dict) -> Tuple[Dict, Optional[Dict]]:
        """
        Generate technical & behavioral interview questions.
        (Left unchanged; uses the LLM and returns usage if API key is set.)
        """
        system_prompt = """You are an experienced technical interviewer. Generate specific interview questions based on the candidate's profile and job requirements.
Return a JSON object with:
{
  "technical_questions": [5 specific questions],
  "behavioral_questions": [3 questions],
  "follow_up_questions": [2 or 3 questions]
}
Focus on skills that both the candidate has and the job requires. Do not include any other fields.
"""

        user_prompt = f"""CANDIDATE PROFILE:
Skills: {', '.join(context.get('candidate_skills', []))}
Experience Level: {context.get('candidate_experience_level', 'Not specified')}
Matched Skills: {', '.join(context.get('matched_skills', context.get('candidate_skills', [])))}

JOB REQUIREMENTS:
Required Skills: {', '.join(context.get('job_skills', []))}
Role Level: {context.get('job_level', 'Not specified')}
Company: {context.get('company', 'Not specified')}

Generate focused interview questions as a JSON object.
"""

        try:
            if not self.client:
                raise RuntimeError("OpenAI client not configured")

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            questions = json.loads(resp.choices[0].message.content)

            # sanitize minimally
            questions.setdefault("technical_questions", [])
            questions.setdefault("behavioral_questions", [])
            questions.setdefault("follow_up_questions", [])
            for key in list(questions.keys()):
                if isinstance(questions[key], list):
                    questions[key] = [str(q) for q in questions[key]]

            return questions, (resp.usage.model_dump() if resp.usage else None)
        except Exception as e:
            return (
                {
                    "error": str(e),
                    "technical_questions": [
                        "Could not generate questions due to an error."
                    ],
                    "behavioral_questions": [
                        "Could not generate questions due to an error."
                    ],
                    "follow_up_questions": [
                        "Could not generate questions due to an error."
                    ],
                },
                None,
            )

    # ========================= Formatting helpers ===========================

    def _format_resume_experience_snippet(self, experience_entries: list, max_chars: int = 200) -> str:
        """
        Use only the first meaningful resume experience snippet (200 chars by default).
        Excludes company and duration on purpose.
        """
        if not experience_entries:
            return "No experience information provided"

        snippet_src = ""
        for exp in experience_entries:
            if isinstance(exp, dict):
                if exp.get("description"):
                    snippet_src = exp["description"].strip()
                    break
                bullets = exp.get("bullets") or []
                if bullets:
                    snippet_src = " ".join(bullets).strip()
                    break
                sentences = exp.get("sentences") or []
                if sentences:
                    snippet_src = " ".join(sentences).strip()
                    break
            else:
                snippet_src = str(exp).strip()
                break

        if not snippet_src:
            return "No experience information provided"

        return snippet_src[:max_chars] + ("..." if len(snippet_src) > max_chars else "")

    def _format_experience_for_llm(self, experience_entries: list, max_entries: int = 4, desc_chars: int = 120) -> str:
        if not experience_entries:
            return "No experience information provided"

        entries = experience_entries[:max_entries]
        formatted = []
        for exp in entries:
            if isinstance(exp, dict):
                parts = []
                if exp.get("title"): parts.append(f"Title: {exp['title']}")
                if exp.get("company"): parts.append(f"Company: {exp['company']}")
                when = exp.get("duration") or f"{exp.get('start_date','?')} - {exp.get('end_date','?')}"
                if when: parts.append(f"Duration: {when}")
                if exp.get("description"):
                    parts.append(f"Description: {exp['description'][:desc_chars]}...")
                if parts: formatted.append(" | ".join(parts))
            else:
                formatted.append(str(exp))
        return " || ".join(formatted)

    def _format_education_for_llm(self, education_entries: list) -> str:
        if not education_entries:
            return "No education information provided"
        formatted = []
        for edu in education_entries:
            if isinstance(edu, dict):
                parts = []
                if edu.get("degree"): parts.append(f"Degree: {edu['degree']}")
                if edu.get("field"): parts.append(f"Field: {edu['field']}")
                if edu.get("institution"): parts.append(f"Institution: {edu['institution']}")
                if edu.get("dates") or edu.get("duration"):
                    parts.append(f"Year: {edu.get('dates') or edu.get('duration')}")
                formatted.append(" | ".join(parts))
            else:
                formatted.append(str(edu))
        return " || ".join(formatted)

    # ============================ Scoring utils ==============================

    def _normalize_skills_list(self, items: List[Any]) -> List[str]:
        """
        Normalize skills list: lowercase, strip punctuation, split common compounds,
        and deduplicate while preserving order.
        """
        if not items:
            return []
        seen = set()
        out: List[str] = []

        def add(token: str):
            token = token.strip()
            if not token: return
            if token not in seen:
                seen.add(token)
                out.append(token)

        for raw in items:
            s = _safe_lower(raw)
            # normalize common patterns
            s = s.replace("&", " and ").replace("/", " / ").replace("+", " + ")
            s = re.sub(r"[^a-z0-9\.\-\+\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()

            # expand a few domain-specific synonyms
            synonyms = {
                "web application firewall": "waf",
                "waf ltm": "waf",
                "ltm": "f5",
                "pcnse": "palo alto pcnse",
                "security plus": "security+",
                "cisco ise": "ise",
                "aruba clearpass": "clearpass",
                "ids ips": "ids / ips",
            }
            for k, v in synonyms.items():
                s = s.replace(k, v)

            # split on " / " to capture ids / ips, etc.
            parts = [p.strip() for p in s.split(" / ")]
            for p in parts:
                add(p)
        return out

    def _semantic_skill_overlap(
        self,
        job_skills: List[str],
        cand_skills: List[str],
        threshold: float = 0.72
    ) -> Tuple[List[str], float]:
        """
        For each job skill, pick the candidate skill with the highest cosine similarity.
        Count as matched if >= threshold OR exact text match. Return unique matched list and score.
        """
        if not job_skills:
            return [], 0.0
        if not cand_skills:
            return [], 0.0

        # Fast exact matches first
        cand_set = set(cand_skills)
        matched = set([s for s in job_skills if s in cand_set])

        # Semantic pass for the rest
        to_check_job = [s for s in job_skills if s not in matched]
        if to_check_job:
            # embed
            job_emb = self.skill_model.encode(to_check_job, convert_to_tensor=True, normalize_embeddings=True)
            cand_emb = self.skill_model.encode(cand_skills, convert_to_tensor=True, normalize_embeddings=True)
            sim = util.cos_sim(job_emb, cand_emb)  # [len(job), len(cand)]

            for i, jskill in enumerate(to_check_job):
                # best candidate similarity
                jrow = sim[i]
                best_idx = int(jrow.argmax())
                best_sim = float(jrow[best_idx])
                if best_sim >= threshold:
                    matched.add(jskill)

        matched_list = sorted(matched, key=lambda x: job_skills.index(x))
        score = len(matched) / float(len(job_skills))
        return matched_list, _clamp01(score)

    def _extract_min_years(self, job_data: Dict) -> int:
        """
        Try to find minimum years from job content (experience list / summary strings).
        Looks for 'minimum X years', 'X+ years', etc.
        """
        buckets = []
        for key in ("experience", "experience_requirements", "requirements", "summary_objective", "description", "education", "education_requirements"):
            val = job_data.get(key)
            if isinstance(val, list):
                buckets.extend([str(x) for x in val])
            elif isinstance(val, dict):
                buckets.extend([str(x) for x in val.values()])
            elif val:
                buckets.append(str(val))

        text = " ".join(buckets).lower()

        # pattern 1: "minimum 5 years"
        m = re.search(r"(?:minimum|min\.?)\s*(\d{1,2})\s+years?", text)
        if m:
            return int(m.group(1))

        # pattern 2: "5+ years"
        m = re.search(r"\b(\d{1,2})\s*\+?\s*years?\b", text)
        if m:
            return int(m.group(1))

        return 0

    def _estimate_candidate_years(self, resume_data: Dict) -> float:
        """
        Estimate candidate total experience in years.
        Strategy:
          1) If summary/objective contains '13 years', use it.
          2) Else, sum durations from experience entries when we can parse.
        """
        # 1) from summary text
        for key in ("summary_objective", "summary", "objective"):
            txt = resume_data.get(key)
            if txt and isinstance(txt, str):
                yrs = self._extract_years_from_text(txt)
                if yrs:
                    return yrs

        # 2) from experience entries
        experience = resume_data.get("experience") or []
        total_months = 0
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            start, end = self._parse_dates_from_entry(exp)
            if start:
                end_dt = end or datetime.utcnow()
                months = (end_dt.year - start.year) * 12 + (end_dt.month - start.month)
                if months > 0:
                    total_months += months

        return round(total_months / 12.0, 2) if total_months > 0 else 0.0

    def _extract_years_from_text(self, text: str) -> Optional[float]:
        """
        Find '13 years', '13+ years', 'over 13 years', etc.
        """
        text = text.lower()
        m = re.search(r"(?:over\s+|more\s+than\s+|plus\s+)?(\d{1,2})\s*\+?\s*years?", text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    def _parse_dates_from_entry(self, exp: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse dates from fields like:
          - "duration": "2022-Jan – Present" or "2014-Mar - 2015-Nov"
          - or "start_date": "2020-Fev", "end_date": "2020-Jul"
        """
        # prefer explicit fields
        start = self._parse_one_date(exp.get("start_date"))
        end = self._parse_one_date(exp.get("end_date"))

        if not start and not end and exp.get("duration"):
            # try to split duration
            dur = str(exp["duration"])
            parts = re.split(r"\s*[–-]\s*", dur)  # en dash or hyphen
            if len(parts) >= 1:
                start = self._parse_one_date(parts[0])
            if len(parts) >= 2:
                end = self._parse_one_date(parts[1])

        return start, end

    def _parse_one_date(self, s: Any) -> Optional[datetime]:
        if not s:
            return None
        s = str(s).strip()
        if not s or s.lower() in ("present", "current", "now"):
            return None
        # expect formats like "2022-Jan" or "2015-Nov"
        m = re.search(r"(\d{4})[/-]?([A-Za-z]{3,9})?", s)
        if not m:
            return None
        year = int(m.group(1))
        mon_txt = (m.group(2) or "jan").lower()
        mon_txt = mon_txt[:3]  # normalize to 3 letters
        month = MONTH_MAP.get(mon_txt, 1)
        try:
            return datetime(year, month, 1)
        except Exception:
            return None

    def _education_match_score(self, resume_data: Dict, job_data: Dict) -> float:
        """
        If job requires Bachelor's (or similar) and resume has any degree -> 1.0.
        If job requires nothing explicit but resume has a degree -> 1.0.
        If no degree on resume -> 0.0.
        """
        job_text = []
        for key in ("education", "education_requirements", "qualification", "qualifications", "summary_objective", "description"):
            val = job_data.get(key)
            if isinstance(val, list):
                job_text.extend([str(x) for x in val])
            elif isinstance(val, dict):
                job_text.extend([str(x) for x in val.values()])
            elif val:
                job_text.append(str(val))
        job_text = " ".join(job_text).lower()

        requires_degree = any(w in job_text for w in ["bachelor", "degree", "bs", "b.sc", "bsc"])
        # Treat engineering diplomas as degree-equivalent
        resume_degrees = self._collect_resume_degrees(resume_data)
        has_degree = len(resume_degrees) > 0

        if requires_degree:
            return 1.0 if has_degree else 0.0
        else:
            # If not explicitly required, but candidate has it, still full score.
            return 1.0 if has_degree else 0.0

    def _collect_resume_degrees(self, resume_data: Dict) -> List[str]:
        degrees = []
        edu = resume_data.get("education") or []
        for e in edu:
            if isinstance(e, dict):
                deg = " ".join([str(e.get("degree") or ""), str(e.get("field") or "")]).strip()
                if deg and len(deg) > 2:
                    degrees.append(deg)
            else:
                s = str(e)
                if len(s) > 2:
                    degrees.append(s)
        # Also scan summary for 'engineer', 'bachelor', etc.
        for key in ("summary_objective", "summary"):
            t = resume_data.get(key)
            if t and re.search(r"\b(engineer|bachelor|licence|masters?|msc|bsc)\b", t.lower()):
                degrees.append("detected_in_summary")
        return degrees

    def _compose_analysis(
        self,
        skills_score: float,
        experience_score: float,
        education_score: float,
        cand_years: float,
        min_years: int,
        matched_skills: List[str],
    ) -> str:
        parts = []
        parts.append(f"Skills overlap is {round(skills_score * 100, 1)}% ({len(matched_skills)} matched).")
        if min_years > 0:
            parts.append(f"Experience meets requirement: {cand_years:g}y vs {min_years}y minimum ({round(experience_score * 100, 1)}%).")
        else:
            if cand_years > 0:
                parts.append(f"Experience provided totals ~{cand_years:g} years.")
            else:
                parts.append("No explicit experience duration found.")
        parts.append(f"Education match: {round(education_score * 100, 1)}%.")
        return " ".join(parts)
