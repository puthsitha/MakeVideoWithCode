"""
Sync Khmer Subtitles (.srt) & HTML Preview Player to actual VoxCPM2 Khmer audio timings.
Ensures every line stays strictly under 40 characters for mobile 9:16 readability.
"""

from __future__ import annotations

import json
from pathlib import Path

TIMINGS_PATH = Path("output/audio_timings.json")
SRT_PATH = Path("subtitles.srt")
HTML_PATH = Path("index.html")

SEGMENT_CUES_KM = {
    "01_intro": [
        "ឪពុកម្តាយស្រឡាញ់អ្នក...",
        "ប៉ុន្តែហេតុអ្វីក្តីសុបិនពួកគាត់\nធ្វើឱ្យអ្នកមានអារម្មណ៍ដូចជាប់ទ្រុង?",
    ],
    "02_verse_1a": [
        "តាំងពីអ្នកមិនទាន់\nស្គាល់ឈ្មោះខ្លួនឯងផង",
        "ព្រួញពីរត្រូវបានកំណត់\nអនាគតអ្នករួចទៅហើយ។",
        "មួយចង្អុលទៅគ្រូពេទ្យ។\nមួយទៀតចង្អុលទៅមេធាវី។",
        "មានសុវត្ថិភាព មានកិត្តិយស\nនិងត្រូវបានគាំទ្រជាស្រេច។",
    ],
    "03_verse_1b": [
        "អស់រយៈពេលជាច្រើនឆ្នាំ\nកតញ្ញូតាធម៌មានន័យថាការយល់ព្រម។",
        "អ្នកមើលឃើញការខិតខំ\nទាំងយប់ទាំងថ្ងៃរបស់ពួកគាត់",
        "ហើយការដឹងគុណក្លាយជា\nខ្សែចងមើលមិនឃើញ...",
        "ទាញដៃអ្នកទៅកាន់ជីវិត\nដែលអ្នកមិនដែលជ្រើសរើស។",
        "ពេលអ្នកដទៃសម្រេចចិត្តធំៗ\nទាំងអស់ជំនួសអ្នក",
        "អ្នកភ្ញាក់ឡើងនៅអាយុម្ភៃប្រាំឆ្នាំ\nមិនដឹងត្រូវសម្រេចចិត្តយ៉ាងណាឡើយ។",
    ],
    "04_hook_midpoint": [
        "សូមសួរដោយស្មោះត្រង់មួយសំណួរ។",
        "តើពួកគាត់កំពុងចិញ្ចឹមមនុស្សម្នាក់\nឬបីអគារការិយាល័យក្នុងភួយទារក?",
        "ព្រោះបើអាជីពរបស់អ្នក\nជាពានរង្វាន់សំណព្វចិត្តពួកគាត់...",
        "តើអ្នកណាជាអ្នកទៅធ្វើការ\nនៅព្រឹកថ្ងៃចន្ទពិតប្រាកដ?",
    ],
    "05_verse_2a": [
        "នេះជាការពិតដែលគ្មាននរណា\nប្រាប់អ្នកនៅតុបាយឡើយ។",
        "ការជ្រើសរើសផ្លូវខ្លួនឯង\nមិនមែនជាការក្បត់ទេ។",
        "ការគោរពការលះបង់របស់ពួកគាត់\nគឺមិនខ្ជះខ្ជាយសេរីភាពអ្នកឡើយ។",
        "ដំបូន្មានពួកគាត់កើតចេញពី\nការភ័យខ្លាចភាពងងឹត",
        "ព្រោះពួកគាត់ធ្លាប់ឆ្លងកាត់\nពិភពលោកគ្មានសំណាញ់សុវត្ថិភាព។",
    ],
    "06_verse_2b": [
        "ដូច្នេះ ឈប់ឈ្លោះក្នុងចិត្ត\nហើយកសាងភស្តុតាងក្រោមពន្លឺថ្ងៃ។",
        "រក្សាសន្តិភាពក្នុងគ្រួសារ\nខណៈពេលអ្នកពង្រឹងជំនាញ",
        "សន្សំប្រាក់ទុករស់បានប្រាំមួយខែ\nហើយបង្ហាញផែនការពិតប្រាកដ",
        "មិនមែនត្រឹមតែក្តីប្រាថ្នាទេ។",
        "ពេលភាពភ័យខ្លាចជួបការត្រៀមខ្លួន\nការស្រឡាញ់នឹងមានកន្លែងដកដង្ហើម។",
    ],
    "07_outro": [
        "អ្នកមានជីវិតតែមួយគត់\nដើម្បីចុះហត្ថលេខាដោយខ្លួនឯង។",
        "សូមប្រាប់ខ្ញុំក្នុងមតិយោបល់ឥឡូវនេះ...",
        "តើអាជីពក្នុងក្តីសុបិនមួយណា\nដែលអ្នកទាមទារសម្រាប់ខ្លួនឯងថ្ងៃនេះ?",
    ],
}


def fmt_ts(sec: float) -> str:
    ms_total = int(round(sec * 1000))
    h = ms_total // 3600000
    m = (ms_total % 3600000) // 60000
    s = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    timings = json.loads(TIMINGS_PATH.read_text())
    entries = []
    idx = 1

    for t in timings:
        sid = t["id"]
        slot_start = float(t["slot_start"])
        offset = float(t.get("offset", 0.0))
        fitted = float(t["fitted_sec"])
        speech_start = slot_start + offset

        cues = SEGMENT_CUES_KM[sid]
        weights = [max(1, len(c.replace("\n", ""))) for c in cues]
        total_w = sum(weights)
        cursor = speech_start

        for i, (cue, w) in enumerate(zip(cues, weights)):
            for line in cue.split("\n"):
                assert len(line) < 40, f"Khmer line >= 40 chars ({len(line)}): {line}"
            dur = fitted * (w / total_w)
            end_t = speech_start + fitted if i == len(cues) - 1 else cursor + dur
            entries.append(f"{idx}\n{fmt_ts(cursor)} --> {fmt_ts(end_t)}\n{cue}")
            cursor = end_t
            idx += 1

    srt_text = "\n\n".join(entries) + "\n"
    SRT_PATH.write_text(srt_text, encoding="utf-8")

    html = HTML_PATH.read_text(encoding="utf-8")
    start_m = "const SRT_CONTENT = `"
    end_m = "`;\n\n    const video ="
    i0 = html.index(start_m) + len(start_m)
    i1 = html.index(end_m, i0)
    html = html[:i0] + srt_text.strip() + html[i1:]
    html = html.replace(
        "<h1>Filial Duty vs Your Own Dreams</h1>",
        "<h1>កាតព្វកិច្ចកូន ឬ ក្តីសុបិនខ្លួនឯង</h1>",
    )
    html = html.replace(
        "<p class=\"subtitle-meta\">Choosing a Career Your Parents Didn&rsquo;t Pick &mdash; Production Preview</p>",
        "<p class=\"subtitle-meta\">ការជ្រើសរើសអាជីពដែលឪពុកម្តាយមិនបានរើសឱ្យ &mdash; Khmer 2.5D + Stick-Man Preview</p>",
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[SRT-KM] Synced {idx - 1} Khmer cues to {SRT_PATH} and {HTML_PATH}.")


if __name__ == "__main__":
    main()
