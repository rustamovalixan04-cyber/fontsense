# Font-name regression images

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
