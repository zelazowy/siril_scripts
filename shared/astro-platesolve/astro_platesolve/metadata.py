"""Astrometric metadata only; applications own source/image/project commits."""
import re
import numpy as np
from astropy.io import fits
from .astap import clean_solution_header, solver_header
from .api import validate_solution

ASTROMETRY_KEY=re.compile(r'(?:WCSAXES|WCSNAME|CTYPE[12]|CUNIT[12]|CRPIX[12]|CRVAL[12]|CDELT[12]|CROTA[12]|(?:CD|PC)[12]_[12]|PV[12]_\d+|(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)|LONPOLE|LATPOLE|RADESYS|RADECSYS|EQUINOX|PLTSOLVD|RA|DEC)')

def merge_headers(original, solved):
    clean=clean_solution_header(solved)
    shape=(int(original.get('NAXIS2',1)),int(original.get('NAXIS1',1)))
    validate_solution(clean,shape)
    result=original.copy()
    for key in list(result):
        if ASTROMETRY_KEY.fullmatch(key) or key in ('CHECKSUM','DATASUM'): del result[key]
    for card in clean.cards:
        if ASTROMETRY_KEY.fullmatch(card.keyword): result.append(card)
    return result


def header_cards(header: str) -> list[str]:
    # Siril's shared-memory header includes its C-string terminator. Keeping
    # END\0 would hide all subsequently appended WCS cards from Siril's parser.
    header = header.partition("\x00")[0]
    if "\n" in header:
        cards = header.splitlines()
    else:
        cards = [header[i:i + 80] for i in range(0, len(header), 80)]
    result = []
    for card in cards:
        if card.strip():
            result.append(card)
        if card[:8].strip() == "END":
            break
    return result


def prepare_solver_header(header: str) -> str:
    """Prevent an undefined Siril equinox sentinel from corrupting the search."""
    cards = header_cards(header)
    for index, card in enumerate(cards):
        if card[:8].strip() != "EQUINOX":
            continue
        try:
            equinox = float(card[10:].split("/", 1)[0].replace("D", "E"))
        except ValueError:
            equinox = float("nan")
        if not np.isfinite(equinox) or not 1000 <= equinox <= 3000:
            cards[index] = "EQUINOX =               2000.0 / Undefined input equinox; use J2000".ljust(80)
    return "\n".join(cards)


