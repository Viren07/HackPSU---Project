"""Find a figure in an open-access paper from a link and a figure label like "Figure 2A".

Source: Europe PMC (free, no key), which mirrors PubMed Central. Steps:
  link -> (pmcid | pmid | doi) -> paper record -> full-text XML (JATS) -> <fig> elements
  -> the requested figure's image + caption.
The caption also tells us what each panel measures ("Panel A shows overall survival..."),
which is used to double-check the outcome read off the figure.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = "PriorArtLens/1.0 (research tool; HackPSU 2026)"
XLINK = "{http://www.w3.org/1999/xlink}href"


class PaperError(Exception):
    """A problem we can explain to the researcher in plain words."""


# ---------------------------------------------------------------- the link
def parse_link(s: str):
    """Return ('pmcid'|'pmid'|'doi', value) from whatever the person pasted."""
    s = urllib.parse.unquote((s or "").strip())
    if not s:
        raise PaperError("Paste a paper link, DOI, PMID or PMC id.")
    m = re.search(r"\bPMC\s?(\d{4,9})\b", s, re.I)
    if m:
        return "pmcid", f"PMC{m.group(1)}"
    m = re.search(r"\b(10\.\d{4,9}/[^\s?#\"<>]+)", s)
    if m:
        doi = re.sub(r"(/full|/abstract|/pdf|/epdf|\.pdf|[.,;)\]])+$", "", m.group(1), flags=re.I)
        return "doi", doi
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", s) or re.search(r"/MED/(\d{5,9})", s, re.I) \
        or re.fullmatch(r"(?:PMID:?\s*)?(\d{5,9})", s, re.I)
    if m:
        return "pmid", m.group(1)
    # Elsevier article code (PII), used by Lancet, Annals of Oncology, Cell Press and ScienceDirect links:
    # S0923-7534(19)34595-8, or the compact form S0923753419345958
    m = re.search(r"(?:PII|(?<![A-Za-z0-9]))(S\d{4}-\d{3}[\dX]\(\d{2}\)\d{5}-[\dX])", s, re.I)
    if m:
        return "pii", m.group(1).upper()
    m = re.search(r"pii/(S\d{7}[\dX]\d{7}[\dX])\b", s, re.I)
    if m:
        c = m.group(1).upper()
        return "pii", f"{c[:5]}-{c[5:9]}({c[9:11]}){c[11:16]}-{c[16]}"
    if re.match(r"https?://", s, re.I):
        return "url", s
    raise PaperError("Could not find a DOI, PMID or PMC id in that link. Try pasting the paper's DOI "
                     "(it starts with 10.) or its PubMed link.")


def parse_figure_spec(s: str):
    """'Figure 2. A', 'Fig 2A', '2a', 'figure 3 panel C' -> (2, 'A'). Panel may be None."""
    t = (s or "").strip()
    if re.search(r"\b(supp|supplement|appendix|^s\d|fig\.?\s*s\d|figure\s*s\d)", t, re.I):
        raise PaperError("Supplementary figures are not part of the paper's main text. "
                         "Download that figure and upload it as an image instead.")
    # a single panel letter right after the number; extra words after it ("C - Squamous") are ignored
    m = re.search(r"(\d{1,2})\s*[.:\-]?\s*(?:panel\s*)?\(?([A-Ha-h])?\)?(?![A-Za-z])", t, re.I)
    if not m:
        raise PaperError("Type the figure like 'Figure 2A' or 'Figure 2'.")
    num = int(m.group(1))
    panel = m.group(2).upper() if m.lastindex and m.lastindex >= 2 and m.group(2) else None
    return num, panel


# ---------------------------------------------------------------- network
def _get(url: str, timeout: int = 25, ua: str = UA) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _search(query: str):
    url = f"{EPMC}/search?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "resultType": "core", "pageSize": 1})
    try:
        data = json.loads(_get(url))
    except Exception as e:
        raise PaperError(f"Could not reach Europe PMC ({e}). Check your internet connection.")
    res = (data.get("resultList") or {}).get("result") or []
    return res[0] if res else None


def _pmid_from_pii(pii: str):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "retmode": "json", "term": f"{pii}[aid]"})
    try:
        ids = json.loads(_get(url))["esearchresult"]["idlist"]
    except Exception as e:
        raise PaperError(f"Could not reach PubMed ({e}). Check your internet connection.")
    return ids[0] if len(ids) == 1 else None


def _ids_from_page(url: str):
    """Last resort for other journal links: read the DOI or PMID from the page's citation tags."""
    try:
        html = _get(url, ua=BROWSER_UA).decode("utf-8", "replace")[:400000]
    except Exception:
        return None
    for pat, kind in ((r'name="citation_pmid"\s+content="(\d{5,9})"', "pmid"),
                      (r'name="(?:citation_doi|dc\.identifier|DC\.Identifier|prism\.doi)"\s+content="(?:doi:)?(10\.[^"]+)"', "doi"),
                      (r'content="(?:doi:)?(10\.\d{4,9}/[^"]+)"\s+name="citation_doi"', "doi")):
        m = re.search(pat, html, re.I)
        if m:
            return kind, m.group(1)
    return None


def resolve(link: str) -> dict:
    kind, val = parse_link(link)
    if kind == "pii":
        pmid = _pmid_from_pii(val)
        if not pmid:
            raise PaperError(f"PubMed has no paper with the article code {val}. Paste the paper's DOI or PubMed link instead.")
        kind, val = "pmid", pmid
    elif kind == "url":
        found = _ids_from_page(val)
        if not found:
            raise PaperError("Could not find a DOI, PMID or PMC id in that link, and the journal page would not tell us. "
                             "Paste the paper's DOI (it starts with 10.) or its PubMed link.")
        kind, val = found
    q = {"pmcid": f"PMCID:{val}", "pmid": f"EXT_ID:{val} AND SRC:MED", "doi": f'DOI:"{val}"'}[kind]
    rec = _search(q)
    if not rec:
        raise PaperError("Europe PMC has no record of that paper. Check the link, or upload the figure as an image.")
    return paper_info(rec)


def paper_info(rec: dict) -> dict:
    title = re.sub(r"<[^>]+>", "", rec.get("title") or "").strip().rstrip(".")
    abstract = re.sub(r"<[^>]+>", " ", rec.get("abstractText") or "")
    authors = rec.get("authorString") or ""
    first = (rec.get("authorList") or {}).get("author") or []
    surname = (first[0].get("lastName") if first else None) or (authors.split(" ")[0] if authors else "")
    year = rec.get("pubYear") or ""
    info = {
        "title": title, "authors": authors, "year": year,
        "journal": (rec.get("journalInfo") or {}).get("journal", {}).get("title") or rec.get("journalTitle") or "",
        "pmid": rec.get("pmid"), "pmcid": rec.get("pmcid"), "doi": rec.get("doi"),
        "open_access": rec.get("isOpenAccess") == "Y",
        "in_pmc": rec.get("inPMC") == "Y" or rec.get("inEPMC") == "Y",
    }
    info["trial_name"] = trial_name(title, abstract) or (f"{surname} {year}".strip() or title[:40])
    info["drugs"] = drugs_from_title(title)
    return info


# ---------------------------------------------------------------- names
_TRIAL_FAMILIES = (r"CheckMate|KEYNOTE|IMpower|IMbrave|IMvigor|ORIENT|RATIONALE|EMPOWER-Lung|JAVELIN|PACIFIC|"
                   r"POSEIDON|CASPIAN|MYSTIC|FLAURA|ALTA|CROWN|POPLAR|OAK|REVEL|LUME-Lung|SQUIRE|TROPION-Lung|"
                   r"CodeBreaK|KRYSTAL|EVOKE|CameL|GEMSTONE|CHOICE|ASTRUM|JUPITER|ORIENT|NCT")
_NOT_TRIALS = {"NSCLC", "SCLC", "OS", "PFS", "HR", "CI", "RCT", "PD-L1", "PD-1", "EGFR", "ALK", "KRAS", "ROS1",
               "TKI", "ICI", "AE", "AES", "ITT", "QOL", "ORR", "DFS", "EFS", "RFS", "UK", "USA", "US", "EU"}


def trial_name(title: str, abstract: str = "") -> str | None:
    text = f"{title} {abstract}"
    m = re.search(rf"\b({_TRIAL_FAMILIES})[\s-]?(\d{{1,4}}[A-Z]?)\b", text)
    if m and not m.group(1).startswith("NCT"):
        return f"{m.group(1)} {m.group(2)}"
    for cand in re.findall(r"\(([A-Za-z][A-Za-z0-9-]{1,18}(?:[ -]\d{1,4})?)\)", title):
        if cand.upper() in _NOT_TRIALS or sum(c.isupper() for c in cand) < 2:
            continue
        return cand
    m = re.search(rf"\b({_TRIAL_FAMILIES})\b", text)
    if m and not m.group(1).startswith("NCT"):
        return m.group(1)
    return None


def drugs_from_title(title: str):
    """'Nivolumab versus Docetaxel in ...' -> {'new': 'Nivolumab', 'old': 'Docetaxel'}."""
    m = re.search(r"([A-Za-z][\w-]+(?:\s+(?:plus|and|\+|with)\s+[A-Za-z][\w-]+)?)\s+(?:versus|vs\.?|compared with)\s+"
                  r"([A-Za-z][\w-]+(?:\s+(?:plus|and|\+)\s+[A-Za-z][\w-]+)?)", title, re.I)
    return {"new": m.group(1), "old": m.group(2)} if m else None


# ---------------------------------------------------------------- outcomes
_EP = [("pfs", r"progression[\s\-–]*free|\bPFS\b"), ("dfs", r"disease[\s\-–]*free|\bDFS\b"),
       ("efs", r"event[\s\-–]*free|\bEFS\b"), ("rfs", r"relapse[\s\-–]*free|recurrence[\s\-–]*free|\bRFS\b"),
       ("os", r"overall\s+survival|\bOS\b")]


def endpoints_in(text: str):
    return [code for code, pat in _EP if re.search(pat, text or "", re.I)]


def panel_text(caption: str, panel: str | None) -> str:
    """The part of a caption about one panel ('Panel A shows ...', '(A) ...', 'A: ...')."""
    if not panel or not caption:
        return ""
    marks = list(re.finditer(r"(?:\bPanels?\s+([A-H])\b|\(([A-H])\)|(?:^|[.;]\s+)([A-H])[:.]\s)", caption))
    for i, m in enumerate(marks):
        if (m.group(1) or m.group(2) or m.group(3)) == panel:
            end = marks[i + 1].start() if i + 1 < len(marks) else len(caption)
            return caption[m.start():end].strip(" ;.")
    return ""


def caption_endpoint(caption: str, panel: str | None):
    """Outcome for the requested panel from the caption; None if unclear."""
    seg = panel_text(caption, panel)
    for text, where in ((seg, f"caption of panel {panel} in the paper"), (caption, "figure caption in the paper")):
        eps = endpoints_in(text)
        if len(eps) == 1:
            return eps[0], where
    return None, None


# ---------------------------------------------------------------- full text
def _clean_xml(raw: bytes) -> str:
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<!DOCTYPE[^>\[]*(\[[^\]]*\])?>", "", s, flags=re.S)       # no entity expansion
    return re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)[A-Za-z][\w.]*;", " ", s)


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def figures_from_xml(raw: bytes):
    root = ET.fromstring(_clean_xml(raw))
    figs = []
    for i, fig in enumerate(root.iter("fig")):
        g = fig.find(".//graphic")
        href = g.get(XLINK) if g is not None else None
        if not href:
            continue
        label = _text(fig.find("label")).rstrip(":. ")
        cap = _text(fig.find("caption"))
        m = re.search(r"(\d{1,2})", label) or re.match(r"\s*(?:Figure|Fig\.?)\s*(\d{1,2})", cap, re.I)
        num = int(m.group(1)) if m else None
        figs.append({"num": num, "label": label or f"Figure {num or i + 1}", "caption": cap, "href": href,
                     "order": i + 1})
    # number figures that carry no label by their order
    if figs and all(f["num"] is None for f in figs):
        for f in figs:
            f["num"] = f["order"]
    return figs


def full_text(pmcid: str) -> bytes:
    urls = [f"{EPMC}/{pmcid}/fullTextXML",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=" + pmcid.replace("PMC", "")]
    for u in urls:
        try:
            raw = _get(u)
            if b"<fig" in raw or b"<article" in raw:
                return raw
        except Exception:
            continue
    raise PaperError("This paper's full text is not available to download, so its figures can't be fetched. "
                     "Take a screenshot of the figure and upload it instead.")


BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


def _pmc_page_images(pmcid: str, href: str):
    base = re.escape(re.sub(r"\.(jpe?g|png|gif|tiff?)$", "", href, flags=re.I))
    pat = re.compile(rf"https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[\w/]+/{base}\.(?:jpe?g|png|gif)", re.I)
    for ua in (UA, BROWSER_UA):
        try:
            html = _get(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/", ua=ua).decode("utf-8", "replace")
            found = list(dict.fromkeys(pat.findall(html)))
            if found:
                return found
        except Exception:
            continue
    return []


def figure_image(pmcid: str, href: str) -> bytes:
    """Download the figure image. Only builds URLs on the two PMC hosts (never a user-supplied URL)."""
    if not re.fullmatch(r"PMC\d{4,9}", pmcid or "") or not re.fullmatch(r"[\w.\-]+", href or ""):
        raise PaperError("Bad figure reference.")
    names = [href] if re.search(r"\.(jpe?g|png|gif|tiff?)$", href, re.I) else [f"{href}.{e}" for e in ("jpg", "png", "gif")]
    # PMC keeps figure images on a CDN under an unguessable folder, so read the exact address
    # from the article page; the older fixed addresses are kept as fallbacks.
    urls = _pmc_page_images(pmcid, href) + \
        [f"https://europepmc.org/articles/{pmcid}/bin/{n}" for n in names] + \
        [f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{n}" for n in names]
    for u in urls:
        try:
            data = _get(u)
            is_image = data[:3] == b"\xff\xd8\xff" or data[:4] in (b"\x89PNG", b"GIF8")
            if len(data) > 2000 and is_image:
                return data
        except Exception:
            continue
    raise PaperError("Found the figure but could not download its image. Take a screenshot and upload it instead.")


def find_figure(link: str, figure: str) -> dict:
    """Everything the page needs: paper details, the figure image and its caption."""
    num, panel = parse_figure_spec(figure)
    info = resolve(link)
    if not info["pmcid"]:
        raise PaperError(f"'{info['title'][:80]}' is not open access, so its figures can't be fetched. "
                         "Take a screenshot of the figure and upload it instead.")
    figs = figures_from_xml(full_text(info["pmcid"]))
    if not figs:
        raise PaperError("No figures found in this paper's full text. Upload the figure as an image instead.")
    match = [f for f in figs if f["num"] == num]
    if not match:
        have = ", ".join(str(f["num"]) for f in figs)
        raise PaperError(f"This paper has no Figure {num}. Its figures are: {have}.")
    fig = match[0]
    ep, ep_where = caption_endpoint(fig["caption"], panel)
    return {
        "paper": info, "panel": panel,
        "figure": {"num": num, "label": fig["label"], "caption": fig["caption"], "href": fig["href"],
                   "panel_text": panel_text(fig["caption"], panel),
                   "endpoint": ep, "endpoint_evidence": ep_where},
        "figures": [{"num": f["num"], "label": f["label"]} for f in figs],
        "image": figure_image(info["pmcid"], fig["href"]),
    }
