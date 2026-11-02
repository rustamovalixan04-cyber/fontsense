# Font-name regression images

`allover_poster_crop.png` is the design region from the user's reported
ALLOVER failure, cropped to exclude the site's controls. It checks separation
of the headline, subtitle and illustration. The user identified the advertised
face as Allover; the publisher's source is
[Allover Modern by 177Studio](https://creativemarket.com/177studio/27125891-Allover-Modern-Font).
It is outside this reference library. The fixture tests correct visible text
and useful serif alternatives, not a fabricated exact-font prediction. No
commercial font program is included.

`ocr_nabla_browser.png` and `ocr_pixelify_sans_browser.png` are two existing
OFL browser development crops of the phrase `Quantum Archive`. They cover an
OCR correction and a case where the second reader is wrong and the original
reading must survive. They were not used as an independent final benchmark.
Source families: [Nabla](https://fonts.google.com/specimen/Nabla) and
[Pixelify Sans](https://fonts.google.com/specimen/Pixelify+Sans). Full capture
provenance remains in the separately supplied browser corpus.

`spaced_headline_poster.png` is a generated mixed-layout regression in the
OFL Bevan font, containing `Cedar Valley` above simple line art. It preserves
a diagnostic case where the detector split two words and selected only the
second. It is a regression fixture, not final accuracy evidence.

`retrieval_encoder_parity.npz` contains 12 OFL-derived image patches and the
actual normalized PyTorch vectors recorded at ONNX export. It verifies that
the portable runtime preserves those outputs within floating-point tolerance;
it is a runtime regression fixture, not accuracy evidence or a font program.

`impact_typography.png` is the complete word Typography rendered locally with
the Windows system Impact font. `impact_screenshot_crop.png` is the cropped text
region from the user's reported failing screenshot. These are regression cases,
not an independent accuracy benchmark. No font program or per-letter bitmap
font is embedded in either PNG.

`times_new_roman_screenshot_crop.png` is the complete-word crop from the user's
next screenshot regression. It checks that Times New Roman/Tinos candidates
remain visible without claiming a verified exact Tinos identity.

Microsoft permits graphics containing words/phrases rendered with Windows fonts;
the font binaries remain local. See the official
[font redistribution FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq).

Impact source information:
[Microsoft Typography](https://learn.microsoft.com/en-us/typography/font-list/impact).
