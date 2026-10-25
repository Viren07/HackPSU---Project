import json

import cv2
import numpy as np
import pytest

from km import paper
from km.paper import (PaperError, caption_endpoint, drugs_from_title, figures_from_xml, parse_figure_spec,
                      parse_link, trial_name)

XML = b"""<?xml version="1.0"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS" "JATS-archivearticle1.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
<fig id="F1"><label>Figure 1</label><caption><title>Trial design&nbsp;overview.</title></caption>
  <graphic xlink:href="nihms-001-f0001"/></fig>
<fig id="F2"><label>Figure 2</label><caption><title>Kaplan&#8211;Meier Curves.</title>
  <p>Panel A shows overall survival in the squamous population. Panel B shows progression-free survival.</p></caption>
  <graphic xlink:href="nihms-001-f0002"/></fig>
</body></article>"""


@pytest.mark.parametrize("s,want", [
    ("https://doi.org/10.1056/NEJMoa1504627", ("doi", "10.1056/NEJMoa1504627")),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa1504627", ("doi", "10.1056/NEJMoa1504627")),
    ("10.1016/S0140-6736(16)32517-X", ("doi", "10.1016/S0140-6736(16)32517-X")),
    ("https://pmc.ncbi.nlm.nih.gov/articles/PMC4681400/", ("pmcid", "PMC4681400")),
    ("https://europepmc.org/article/MED/26028407", ("pmid", "26028407")),
    ("https://pubmed.ncbi.nlm.nih.gov/26028407/", ("pmid", "26028407")),
    ("PMID: 26028407", ("pmid", "26028407")),
    ("https://www.annalsofoncology.org/article/S0923-7534(19)34595-8/fulltext", ("pii", "S0923-7534(19)34595-8")),
    ("https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(16)32517-X/fulltext", ("pii", "S0140-6736(16)32517-X")),
    ("https://www.sciencedirect.com/science/article/pii/S0923753419345958", ("pii", "S0923-7534(19)34595-8")),
    ("https://ascopubs.org/some/article/page", ("url", "https://ascopubs.org/some/article/page")),
])
def test_parse_link(s, want):
    assert parse_link(s) == want


def test_pii_to_pmid(monkeypatch):
    calls = []

    def fake_get(url, timeout=25, ua=None):
        calls.append(url)
        if "esearch" in url:
            return json.dumps({"esearchresult": {"idlist": ["29045553"]}}).encode()
        if "/search?" in url:
            return json.dumps({"resultList": {"result": [{"title": "X versus Y in Z", "pmid": "29045553"}]}}).encode()
        raise OSError("404")
    monkeypatch.setattr(paper, "_get", fake_get)
    info = paper.resolve("https://www.annalsofoncology.org/article/S0923-7534(19)34595-8/fulltext")
    assert info["pmid"] == "29045553" and "S0923-7534%2819%2934595-8%5Baid%5D" in calls[0]
    assert "EXT_ID%3A29045553" in calls[1]


def test_doi_from_journal_page(monkeypatch):
    page = b'<meta name="citation_doi" content="10.1200/JCO.2020.38.1">'
    monkeypatch.setattr(paper, "_get", lambda url, timeout=25, ua=None: page)
    assert paper._ids_from_page("https://ascopubs.org/x") == ("doi", "10.1200/JCO.2020.38.1")


@pytest.mark.parametrize("s,want", [("Figure 2-A", (2, "A")), ("Figure 2. A", (2, "A")), ("Fig 2A", (2, "A")), ("2a", (2, "A")),
                                    ("Figure 3 panel C", (3, "C")), ("Figure 3 C - Squamous", (3, "C")), ("Figure 3 Squamous", (3, None)), ("Figure 2", (2, None)), ("fig. 4(B)", (4, "B"))])
def test_parse_figure_spec(s, want):
    assert parse_figure_spec(s) == want


def test_supplementary_rejected():
    with pytest.raises(PaperError):
        parse_figure_spec("Figure S2")


def test_names():
    assert trial_name("Atezolizumab versus docetaxel in patients with previously treated non-small-cell lung "
                      "cancer (OAK): a phase 3, open-label, multicentre randomised controlled trial") == "OAK"
    assert trial_name("Nivolumab versus Docetaxel in Advanced Squamous-Cell NSCLC",
                      "In this trial (CheckMate 017), we ...") == "CheckMate 017"
    assert trial_name("Pembrolizumab in NSCLC (KEYNOTE-010)") == "KEYNOTE 010"
    assert trial_name("A study of something in (NSCLC)") is None
    assert drugs_from_title("Nivolumab versus Docetaxel in Advanced Squamous-Cell Lung Cancer") == \
        {"new": "Nivolumab", "old": "Docetaxel"}


def test_caption_panels():
    cap = "Panel A shows overall survival. Panel B shows progression-free survival."
    assert caption_endpoint(cap, "A")[0] == "os"
    assert caption_endpoint(cap, "B")[0] == "pfs"
    assert caption_endpoint(cap, None)[0] is None          # both outcomes, no panel: unclear
    lancet = "Figure 2: Overall survival (A) ITT population. (B) Squamous. (C) Non-squamous."
    assert caption_endpoint(lancet, "C")[0] == "os"


def test_figures_from_xml():
    figs = figures_from_xml(XML)
    assert [f["num"] for f in figs] == [1, 2]
    assert figs[1]["href"] == "nihms-001-f0002" and "Panel A" in figs[1]["caption"]


def test_find_figure_offline(monkeypatch):
    img = np.full((400, 600, 3), 255, np.uint8)
    png = cv2.imencode(".jpg", img)[1].tobytes()
    rec = {"title": "Nivolumab versus Docetaxel in Advanced Squamous-Cell Non-Small-Cell Lung Cancer",
           "authorString": "Brahmer J, Reckamp KL, Baas P", "pubYear": "2015", "pmcid": "PMC4681400",
           "isOpenAccess": "N", "inEPMC": "Y", "abstractText": "...", "authorList": {"author": [{"lastName": "Brahmer"}]}}

    cdn = "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/0392/4681400/2715ebdc0d2f/nihms-001-f0002.jpg"

    def fake_get(url, timeout=25, ua=None):
        if "/search?" in url:
            return json.dumps({"resultList": {"result": [rec]}}).encode()
        if "fullTextXML" in url:
            return XML
        if url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC4681400/":
            return f'<img src="{cdn.replace("f0002", "f0001")}"><img src="{cdn}">'.encode()
        if url == cdn:
            return png
        raise OSError("404")
    monkeypatch.setattr(paper, "_get", fake_get)
    r = paper.find_figure("https://doi.org/10.1056/NEJMoa1504627", "Figure 2. A")
    assert r["paper"]["trial_name"] == "Brahmer 2015"
    assert r["paper"]["drugs"] == {"new": "Nivolumab", "old": "Docetaxel"}
    assert r["figure"]["endpoint"] == "os" and r["panel"] == "A"
    assert r["image"] == png
    with pytest.raises(PaperError, match="no Figure 5"):
        paper.find_figure("PMC4681400", "Figure 5")
