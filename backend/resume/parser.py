"""Regex-based resume text extraction for the educational system."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
SECTION_NAMES = {
    "education": ("education", "academic background", "academics"),
    "skills": ("skills", "technical skills", "technologies"),
    "projects": ("projects", "academic projects", "personal projects"),
    "experience": ("experience", "work experience", "employment", "internships", "internship"),
    "certifications": ("certifications", "certificates", "licenses"),
    "achievements": ("achievements", "awards", "honors"),
}
SKILL_TERMS = (
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "flask", "fastapi",
    "react", "node.js", "mongodb", "mysql", "postgresql", "opencv", "pytorch", "tensorflow",
    "docker", "aws", "git", "linux", "html", "css", "machine learning", "deep learning",
)
DEGREE_RE = re.compile(r"\b(?:B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|BSc|MSc|Bachelor|Master|Ph\.?D)\b[^\n]*", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
BULLET_RE = re.compile(r"^\s*[\u2022*+\-]\s*(.+)$")


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def extract_sections(text: str) -> dict[str, list[str]]:
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    sections: dict[str, list[str]] = {name: [] for name in SECTION_NAMES}
    current: str | None = None
    aliases = {alias.lower(): name for name, values in SECTION_NAMES.items() for alias in values}
    for line in lines:
        header = line.rstrip(":-").strip().lower()
        if header in aliases:
            current = aliases[header]
            continue
        if current:
            sections[current].append(line)
    return sections


def extract_resume(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("resume text must be non-empty")
    cleaned = normalize_text(text)
    sections = extract_sections(cleaned)
    skills_text = "\n".join(sections["skills"]).lower()
    skills = sorted({term for term in SKILL_TERMS if re.search(rf"(?<![\w+#]){re.escape(term.lower())}(?![\w+#])", skills_text)})
    projects = _clean_items(sections["projects"])
    experience = _clean_items(sections["experience"])
    education = _clean_items(sections["education"])
    certifications = _clean_items(sections["certifications"])
    return {
        "contact": {
            "emails": sorted(set(EMAIL_RE.findall(cleaned))),
            "phones": sorted(set(_normalize_phone(match) for match in PHONE_RE.findall(cleaned))),
            "urls": sorted(set(URL_RE.findall(cleaned))),
        },
        "education": education or _find_matches(DEGREE_RE, cleaned),
        "skills": skills,
        "projects": projects,
        "experience": experience,
        "certifications": certifications,
        "years": sorted(set(YEAR_RE.findall(cleaned))),
        "sections_found": [name for name, values in sections.items() if values],
    }


def _clean_items(lines: list[str]) -> list[str]:
    items = []
    for line in lines:
        match = BULLET_RE.match(line)
        value = match.group(1) if match else line
        if value and value not in items:
            items.append(value)
    return items


def _find_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).strip() for match in pattern.finditer(text)))


def _normalize_phone(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def extract_resume_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    elif suffix == ".docx":
        from docx import Document
        text = "\n".join(paragraph.text for paragraph in Document(file_path).paragraphs)
    else:
        raise ValueError("supported resume formats are PDF, DOCX, and TXT")
    return extract_resume(text)
