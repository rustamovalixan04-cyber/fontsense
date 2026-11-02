"""Deterministic, renamed Latin reference subsets of verified OFL sources.

This module runs during builds only. Keep copyright/license names; rename all
font-family/PostScript names because OFL sources can have Reserved Font Names.
No Windows or other proprietary font program is processed here.
"""
from io import BytesIO
import hashlib

from fontTools import subset
from fontTools.ttLib import TTFont


def make_subset(data, source_id):
    font = TTFont(BytesIO(data), recalcTimestamp=False)
    cmap = font.getBestCmap()
    if not all(ord(char) in cmap for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
        raise ValueError("Reference lacks the basic Latin alphabet")
    # Copyright/trademark/license IDs and variation-axis labels stay.
    for name in list(font["name"].names):
        if name.nameID in (1, 2, 3, 4, 6, 16, 17, 18, 20, 21, 22, 25):
            font["name"].names.remove(name)
    family = "FSReference" + hashlib.sha256(source_id.encode()).hexdigest()[:16]
    for platform, encoding, language in ((3, 1, 0x409), (1, 0, 0)):
        for name_id, value in ((1, family), (2, "Regular"), (3, family + "-1"), (4, family),
                               (6, family), (16, family), (17, "Regular")):
            font["name"].setName(value, name_id, platform, encoding, language)
    options = subset.Options()
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]
    options.name_legacy = True
    options.name_languages = ["*"]
    # Shaping tables and variation axes are retained for real font rendering.
    worker = subset.Subsetter(options=options)
    worker.populate(unicodes=range(32, 256))
    worker.subset(font)
    output = BytesIO()
    font.save(output, reorderTables=True)
    font.close()
    return output.getvalue()
