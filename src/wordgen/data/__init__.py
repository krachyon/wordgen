from pathlib import Path

english_words = Path(__file__).parent/"english_words.txt"
german_words = Path(__file__).parent/"german_words.txt"
finnish_words = Path(__file__).parent/"finnish_words.txt"
eng_wik = Path(__file__).parent/"eng_wiki.txt"
deu_mixed = Path(__file__).parent/"deu_mixed.txt"
fin_wiki = Path(__file__).parent/"fin_wiki.txt"
nld_mixed = Path(__file__).parent/"nld_mixed.txt"
finnish_hyp = Path(__file__).parent/"hyph_fi_FI.dic"

def all_text() -> list[Path]:
    return list(Path(__file__).parent.glob("*.txt"))