# Topic 1: ដើមខែសេដ្ឋី ចុងខែមីកញ្ចប់ — Connected Gemini Video Prompt Pack (`scene_00_hook` → `scene_08_outro`)

- **Target Engine:** Gemini Video (**Gemini Omni 1.2 Flash** / **Veo 3** First-Frame-to-Video)
- **Format:** `9:16` vertical (`1080×1920`, `24fps`/`30fps`)
- **Total Duration:** **75 seconds** (`0:00 – 1:15`) across **13 connected scenes**
- **Style Lock:** DEMIC-inspired 2D flat-vector cel-shaded animation, crisp dark-brown outlines, snappy cartoon physics, **no spoken dialogue**, **no on-screen text** (Khmer captions are overlaid cleanly afterward via `render_topic1_reel.py`)
- **Batch JSON (ready for `generate_video.py --batch`):** [topic1_gemini_video_prompts.json](./topic1_gemini_video_prompts.json)

---

## Master Timeline & Scene-to-Scene Connection Map

| # | Scene ID | Starting Frame | Timecode | Duration | Transition Bridge (OUT → Next Scene) |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 0 | `scene_00_hook` | `scene_00_hook.jpg` | `0:00–0:03` | **3s** | Top golden glow intensifies & punches in → bursts into `scene_01` golden sunburst |
| 1 | `scene_01_payday_khqr` | `scene_01_payday_khqr.jpg` | `0:03–0:09` | **6s** | Bright white-gold QR scan flash peaks & whips right → carries into `scene_02a` strut |
| 2 | `scene_02a_iced_coffee` | `scene_02a_iced_coffee.jpg` | `0:09–0:12` | **3s** | Strut accelerates right with diagonal speed lines whip-pan → lands in `scene_02b` amber BBQ |
| 3 | `scene_02b_treat_bbq` | `scene_02b_treat_bbq.jpg` | `0:12–0:15` | **3s** | Phone screen flashes with a bright sparkle & camera pushes toward phone → matches `scene_02c` bed phone |
| 4 | `scene_02c_delivery_parcels` | `scene_02c_delivery_parcels.jpg` | `0:15–0:19` | **4s** | Giant cardboard box slams down hard at `4.0s` → hard smash-cut + record scratch into `scene_03` cold grey |
| 5 | `scene_03_day12_shock` | `scene_03_day12_shock.jpg` | `0:19–0:26` | **7s** | Cold grey-blue shadows darken & camera drifts down → settles into `scene_04a` dim bedroom |
| 6 | `scene_04a_airplane_mode` | `scene_04a_airplane_mode.jpg` | `0:26–0:30` | **4s** | Character sinks deeper under grey blanket → cool grey cross-fades into `scene_04b` pale grey table |
| 7 | `scene_04b_coffee_downgrade` | `scene_04b_coffee_downgrade.jpg` | `0:30–0:34` | **4s** | Room dims around him into an overhead spotlight cone → matches `scene_04c` dark spotlight table |
| 8 | `scene_04c_noodles_bbq_photo` | `scene_04c_noodles_bbq_photo.jpg` | `0:34–0:39` | **5s** | Single tear drops at tragic peak & sad violin stops dead → explosive snap-cut into `scene_05` mint sunburst |
| 9 | `scene_05_lightbulb_idea` | `scene_05_lightbulb_idea.jpg` | `0:39–0:44` | **5s** | Golden lightbulb glow & mint-white sunburst rays sweep outward → seamless match into `scene_06` mint sunburst |
| 10 | `scene_06_three_jars` | `scene_06_three_jars.jpg` | `0:44–0:59` | **15s** | Golden padlock snaps shut & camera pushes softly toward glowing green jar → match-dissolves to `scene_07` sprouted green jar |
| 11 | `scene_07_new_day25` | `scene_07_new_day25.jpg` | `0:59–1:08` | **9s** | Warm mint-cream sunlight blooms softly & camera pulls back → clears upper space into `scene_08` outro |
| 12 | `scene_08_outro` | `scene_08_outro.jpg` | `1:08–1:15` | **7s** | Calm gentle sparkle hold + 2s warm musical fade-out → clean Reels loop back to `scene_00` |

> **Gemini Duration Notes:**
> - **Gemini Omni 1.2 Flash (`gemini-omni-1.2-flash`)** natively supports exact integer durations from **3s to 15s**, so every single scene above (`3s` up to `15s`) generates at its **exact target duration** without trimming!
> - **Gemini Web / Veo 3 (fixed 8s clips):** Paste the same prompt; for shorter scenes (`3s–7s`), trim to the exact end timestamp in editing; for `scene_06_three_jars` (**15s**), generate an 8s clip covering `[0.0s–8.0s]` (Blue + Orange jars) and extend by 7s covering `[8.0s–15.0s]` (Green jar + padlock lock + push-in).
> - **Character Consistency:** Attach `scene_<id>.jpg` as the **starting frame** (`--first-frame`). Optionally attach `khmer_reel_character_sheet.jpg` as a secondary reference image if needed, though each `scene_<id>.jpg` already locks the exact character and framing.

---

## All 13 Connected Gemini Video Prompts

### Scene 0 — `scene_00_hook` (`0:00 – 0:03` | Duration: **3s**)
- **Input Image:** `topic1_scenes/scene_00_hook.jpg`
- **Connects From:** Reels opening hook (instant visual contrast)
- **Connects To:** `scene_01_payday_khqr` (top golden-yellow flash & punch-in bursts into payday sunburst)
- **Khmer Overlay (added in post):** `ដើមខែសេដ្ឋី` (top) / `ចុងខែមីកញ្ចប់ 😂` (bottom)

```text
Use the attached image (scene_00_hook.jpg) as the starting frame. Duration: 3 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and snappy cartoon physics, no random scene cuts. Maintain the horizontal split-screen divided by the bold dark-brown center bar. [0.0s–1.2s] On the warm golden-yellow top half, the young Cambodian man in black sunglasses smirks proudly and briskly fans his five green banknotes as tiny yellow sparkles twinkle, his sunglasses glint diagonally, and white steam rises energetically off the sizzling tabletop BBQ meat. [1.2s–2.4s] Simultaneously on the cold grey-blue bottom half, the same man hunched over his bowl of instant noodles blinks slowly in misery as thick comic blue waterfall tears stream continuously down both cheeks, his chopsticks dip weakly into the noodles with faint wisps of steam, and the propped smartphone beside the bowl glows brightly with its BBQ photo. [2.4s–3.0s] Outgoing transition beat: the top half's golden-yellow background brightens with an energetic slight punch-in zoom and sparkling flash on the banknotes to burst seamlessly into the next payday sunburst scene. Sound design: punchy upbeat funk bass groove starts immediately with a crisp impact thump and sizzling BBQ grill on the first beat, followed by a comical sad noodle slurp at 1.6s. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks. Keep caption zones clean.
```

---

### Scene 1 — `scene_01_payday_khqr` (`0:03 – 0:09` | Duration: **6s**)
- **Input Image:** `topic1_scenes/scene_01_payday_khqr.jpg`
- **Connects From:** `scene_00_hook` (catches the golden-yellow energy burst from Scene 0's top half)
- **Connects To:** `scene_02a_iced_coffee` (white-gold QR scan flash peaks and whips right into the iced-coffee strut)
- **Khmer Overlay (added in post):** `ថ្ងៃទី ១ — ប្រាក់ខែចូល! 💸` / `ស្កេនគ្មានសួរតម្លៃ! 😎`

```text
Use the attached image (scene_01_payday_khqr.jpg) as the starting frame. Duration: 6 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and snappy ease-out cartoon physics, no random scene cuts. Opening directly out of the previous scene's warm golden glow as the yellow sunburst rays rotate gently behind the character. [0.0s–1.8s] The young Cambodian man stands proudly at the café counter with eyes happily closed in a smug grin; the tiny golden cartoon crown drops gently onto his head with a playful spring bounce while green banknotes and golden four-point sparkles drift upward around him. [1.8s–3.8s] With swaggering cowboy fast-draw energy, he whips his right arm and smartphone briskly forward toward the red-and-blue tabletop QR standee on the counter, triggering a bright spiky white-and-yellow scan flash burst between phone and standee. [3.8s–5.3s] Extra sparkles and floating banknotes pop outward from the flash as he nods smugly with shoulders bouncing to the beat. [5.3s–6.0s] Outgoing transition beat: the bright golden scan flash expands across the right side of frame and initiates a snappy rightward whip-pan energy leading directly into the next walking scene. Sound design: confident bouncy upbeat funk groove continues seamlessly; playful pop and cash-register cha-ching at 0.4s, cowboy whip-whoosh at 2.2s, sharp digital QR scan beep at 2.6s, and celebratory sparkle pop at 3.2s. Keep top and bottom areas clean for captions. Strictly no spoken dialogue, no voiceover, no readable text or letters on the QR standee, no numbers, no logos, no watermarks.
```

---

### Scene 2A — `scene_02a_iced_coffee` (`0:09 – 0:12` | Duration: **3s**)
- **Input Image:** `topic1_scenes/scene_02a_iced_coffee.jpg`
- **Connects From:** `scene_01_payday_khqr` (catches the rightward whip-pan momentum out of Scene 1's QR scan)
- **Connects To:** `scene_02b_treat_bbq` (accelerates off-right with diagonal speed lines whip-panning into the BBQ dinner)
- **Khmer Overlay (added in post):** None (clean fast visual beat)

```text
Use the attached image (scene_02a_iced_coffee.jpg) as the starting frame. Duration: 3 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and buoyant 2D walk-cycle physics, no random scene cuts. Entering swiftly on rightward whip-pan momentum from the previous café scene onto the warm peach-yellow background. [0.0s–1.8s] Wearing black sunglasses with a cool smug grin, the young Cambodian man in his pastel-blue shirt, dark navy trousers, and white sneakers performs a confident rhythmic full-body strut toward the right, head bobbing playfully to the beat, sunglasses catching a crisp diagonal white glint, while proudly raising his tall iced milk coffee cup so the whipped cream and green straw bounce subtly and golden sparkles twinkle around him. [1.8s–2.5s] Diagonal orange speed lines streak briskly across the background as his stride picks up buoyant momentum. [2.5s–3.0s] Outgoing transition beat: he struts eagerly toward the right edge as the camera performs a fast rightward whip-pan blur into warm golden-amber light for the next dinner scene. Sound design: upbeat funk groove climbing in tempo, crisp iced-coffee straw slurp at 0.35s, and a snappy rightward whip-pan whoosh at 2.65s. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 2B — `scene_02b_treat_bbq` (`0:12 – 0:15` | Duration: **3s**)
- **Input Image:** `topic1_scenes/scene_02b_treat_bbq.jpg`
- **Connects From:** `scene_02a_iced_coffee` (lands directly out of Scene 2A's rightward whip-pan into cozy evening amber)
- **Connects To:** `scene_02c_delivery_parcels` (phone screen sparkles brightly and camera pushes gently toward the phone, bridging to Scene 2C's glowing bed phone)
- **Khmer Overlay (added in post):** `ប៉ាវមួយតុ! 🍖`

```text
Use the attached image (scene_02b_treat_bbq.jpg) as the starting frame. Duration: 3 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Landing directly from the previous rightward whip-pan into the warm evening amber-orange Cambodian BBQ restaurant interior with glowing string lights. [0.0s–1.6s] Seated proudly at the center behind the sizzling tabletop BBQ hotpot grill piled with meat and veggies, the young Cambodian man waves his raised left hand generously like a boss while holding his smartphone proudly in his right hand; both his male friend on the left and female friend on the right pump their fists and cheer excitedly with bouncy celebratory motion. [1.6s–2.4s] Cartoon white steam puffs swirl upward from the grill and soup pot while pink hearts and cute smiling yellow stars float happily upward around him. [2.4s–3.0s] Outgoing transition beat: the character glances proudly at his smartphone as a bright golden tap-sparkle flashes on the screen and the camera pushes slightly inward toward the glowing phone, matching the glowing phone focus of the next bedroom scene. Sound design: energetic upbeat funk groove continues, bright pop at 0.15s, lively happy crowd cheers, and warm sizzling BBQ grill audio throughout. Keep upper and lower margins clean for captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 2C — `scene_02c_delivery_parcels` (`0:15 – 0:19` | Duration: **4s**)
- **Input Image:** `topic1_scenes/scene_02c_delivery_parcels.jpg`
- **Connects From:** `scene_02b_treat_bbq` (picks up from Scene 2B's glowing phone sparkle at peak spending euphoria)
- **Connects To:** `scene_03_day12_shock` (final heavy box slams down at `4.0s` triggering a hard smash-cut + record scratch into icy grey-blue shock)
- **Khmer Overlay (added in post):** `ដឹកដល់ផ្ទះ! 📦`

```text
Use the attached image (scene_02c_delivery_parcels.jpg) as the starting frame. Duration: 4 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and snappy squash-and-stretch cartoon physics, no random scene cuts. Picking up seamlessly from the glowing smartphone energy of the previous scene in this warm pastel-peach bedroom. [0.0s–1.8s] Lying happily back on his bed with sparkling starry eyes and a wide excited grin, the young Cambodian man taps his glowing smartphone rapidly as golden sparkles burst continuously from the screen while two tiny cartoon delivery drivers on scooters zoom briskly from left to right across the background with horizontal speed lines. [1.8s–3.4s] A comical cascade of plain blank brown taped cardboard delivery boxes rains steadily down from the top of the frame with vertical motion lines, landing with bouncy squish impacts and tiny white dust puffs as the mountain of parcels stacks higher around the bed. [3.4s–4.0s] Outgoing transition beat: the pace hits peak comic frenzy as one oversized cardboard box drops swiftly from above and slams down hard right at 3.95s with a sharp impact frame, setting up an abrupt smash-cut into freezing shock in the next scene. Sound design: upbeat funk groove reaches peak tempo, playful mini-scooter vroom at 0.15s, rapid ascending pop-pop-pop parcel drops from 1.0s to 3.2s, ending on a sudden heavy thump at 3.95s. All cardboard boxes remain strictly plain brown with tape only. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 3 — `scene_03_day12_shock` (`0:19 – 0:26` | Duration: **7s**)
- **Input Image:** `topic1_scenes/scene_03_day12_shock.jpg`
- **Connects From:** `scene_02c_delivery_parcels` (hard smash-cut from warm peach frenzy into cold muted grey-blue silence)
- **Connects To:** `scene_04a_airplane_mode` (cold grey-blue shadows darken and camera drifts downward into the dim night bedroom)
- **Khmer Overlay (added in post):** `ថ្ងៃទី ១២ សោះ?! 😱` / `សល់ ២,៥០០៛`

```text
Use the attached image (scene_03_day12_shock.jpg) as the starting frame. Duration: 7 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Hard smash-cut arrival out of the previous warm scene into this sudden cold muted grey-blue background with slanted dark rain-like gloom lines. [0.0s–0.4s] Quick subtle camera impact jolt as the character freezes stiff in comic panic, holding his smartphone in his right hand while an ominous red glow radiates onto his arm and pastel-blue shirt. [0.4s–2.2s] Comic dead-silence beat: the character trembles slightly and slowly parts two fingers of his left hand covering his face to peek nervously at the glowing red phone screen as the red light pulses menacingly. [2.2s–4.8s] His eyes widen in stunned horror, his jaw drops slightly, the large glossy blue cartoon sweat drop slides slowly down his cheek, and the tiny translucent white cartoon ghost soul wiggles and floats gently upward out of the top of his head. [4.8s–6.3s] Slanted gloomy background lines drift downward slowly as his shoulders sag in defeated shock. [6.3s–7.0s] Outgoing transition beat: the cold grey-blue lighting dims slightly darker and the camera drifts subtly downward, seamlessly matching the dark grey-blue night bedroom of the next scene. Sound design: sharp vinyl record scratch at 0.0s cutting all music into 2 seconds of complete dead silence (0.3s–2.2s) for comedic punchline, then a flat error buzz at 2.25s, a comical ghostly wooo at 3.1s, and a soft eerie low drone fading into the end. Leave generous empty space at top and bottom for Khmer captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 4A — `scene_04a_airplane_mode` (`0:26 – 0:30` | Duration: **4s**)
- **Input Image:** `topic1_scenes/scene_04a_airplane_mode.jpg`
- **Connects From:** `scene_03_day12_shock` (continues directly from Scene 3's darkening cold grey-blue dread)
- **Connects To:** `scene_04b_coffee_downgrade` (character pulls blanket over head and cool grey palette cross-fades to Scene 4B's pale grey backdrop)
- **Khmer Overlay (added in post):** `ល្ងាចនេះជុំគ្នា?` / `បិទសេវាសិន ✈️`

```text
Use the attached image (scene_04a_airplane_mode.jpg) as the starting frame. Duration: 4 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Continuing directly from the previous scene's cold grey-blue gloom into this dim nighttime bedroom. [0.0s–1.6s] Hiding under his grey blanket on the bed with wide sweating panicked eyes peeking over the edge, the young Cambodian man shivers nervously as the smartphone flat on the bed glows upward and the four blank red chat-notification bubbles bounce and pulse urgently around it. [1.6s–2.6s] His right index finger lunges frantically to tap the large glowing blue circular airplane-mode icon hovering above the phone; the blue icon flashes brightly on impact and all four red chat bubbles instantly pop and vanish into tiny puff rings. [2.6s–3.5s] He exhales nervously and pulls the grey blanket higher around his shoulders, shrinking deeper into hiding. [3.5s–4.0s] Outgoing transition beat: the dim cool grey tones soften evenly across the frame, preparing a smooth melancholy cross-fade into the pale grey table scene next. Sound design: slow comedic sad piano enters softly at 0.0s, two urgent chat notification pings at 0.25s and 0.95s, and a crisp decisive tap pop at 2.0s as the bubbles disappear. Keep top and bottom margins clear for captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 4B — `scene_04b_coffee_downgrade` (`0:30 – 0:34` | Duration: **4s**)
- **Input Image:** `topic1_scenes/scene_04b_coffee_downgrade.jpg`
- **Connects From:** `scene_04a_airplane_mode` (continues the muted cool-grey sadness and slow sad piano)
- **Connects To:** `scene_04c_noodles_bbq_photo` (background dims into an overhead spotlight cone matching Scene 4C's dramatic spotlight)
- **Khmer Overlay (added in post):** None (pure visual comedy gag)

```text
Use the attached image (scene_04b_coffee_downgrade.jpg) as the starting frame. Duration: 4 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Cross-fading cleanly from the previous scene's cool grey palette onto this pale desaturated grey background. Behind the wooden table, the young Cambodian man sits slumped resting his chin heavily in both hands with tired, miserable half-lidded eyes. [0.0s–1.2s] Camera subtly glides left-to-right across the tabletop progression: the faded tall fancy iced milk coffee cup on the left turns even paler and dims out with a tiny comic puff as the first rightward arrow pulses softly. [1.2s–2.4s] Focus shifts to the middle classic Southeast Asian iced coffee in a tied plastic bag with straw, which shrinks slightly with a second gentle cartoon poof as the second rightward arrow highlights the small plain glass of clear water on the right. [2.4s–3.4s] A tiny lonely sparkle glints off the plain water glass while the man lets out a heavy defeated sigh, shoulders slumping slightly lower as his eyes stay fixed miserably on the water. [3.4s–4.0s] Outgoing transition beat: the pale grey background gradually dims at the edges into a soft dark overhead spotlight cone, bridging seamlessly into the dark spotlight noodle scene next. Sound design: slow comedic sad piano joined by a weeping melancholic sad violin, two soft cartoon poof sounds at 1.0s and 2.15s, followed by a quiet deflated sigh. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 4C — `scene_04c_noodles_bbq_photo` (`0:34 – 0:39` | Duration: **5s**)
- **Input Image:** `topic1_scenes/scene_04c_noodles_bbq_photo.jpg`
- **Connects From:** `scene_04b_coffee_downgrade` (arrives seamlessly inside the dim overhead spotlight cone from Scene 4B)
- **Connects To:** `scene_05_lightbulb_idea` (tragic tear drops and sad violin cuts dead at `5.0s`, priming Scene 5's explosive mint sunburst snap-cut)
- **Khmer Overlay (added in post):** `មើលរូបសាច់អាំង ក្លែមមីកញ្ចប់ 😭`

```text
Use the attached image (scene_04c_noodles_bbq_photo.jpg) as the starting frame. Duration: 5 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Continuing smoothly inside the dim triangular overhead spotlight cone on a dark gloomy grey-blue background from the previous scene. Seated in profile on the small wooden stool at the bare table, the young Cambodian man hunches miserably over his small white bowl of plain instant noodles, with the torn silver wrapper holding the saved half-block of uncooked dry noodles beside it. [0.0s–1.6s] A thin delicate wisp of steam curls slowly up from the noodle bowl as he pokes the noodles half-heartedly with his chopsticks while staring longingly at the propped smartphone glowing warmly with a vivid photo of sizzling grilled BBQ meat and skewers. [1.6s–3.3s] He leans closer toward the glowing BBQ photo on the phone screen, closes his eyes wistfully, and takes a slow dramatic deep sniff as if smelling real grilled meat. [3.3s–4.5s] Opening his sad eyes, the single glossy blue cartoon tear detaches and rolls slowly down his cheek, dropping gently off his chin as the camera holds in pathetic comic stillness. [4.5s–5.0s] Outgoing transition beat: the scene hits its peak dramatic gloom and holds a crisp half-second comic pause right before an abrupt upbeat reversal in the next scene. Sound design: weeping sad violin peaks emotionally with slow piano chords, subtle dry noodle rustle at 0.45s, comical deep dramatic sniff at 2.3s, soft glistening teardrop plop at 3.4s, then music abruptly cuts off right at 4.95s. Keep top and bottom areas clean for captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 5 — `scene_05_lightbulb_idea` (`0:39 – 0:44` | Duration: **5s**)
- **Input Image:** `topic1_scenes/scene_05_lightbulb_idea.jpg`
- **Connects From:** `scene_04c_noodles_bbq_photo` (explosive snap-cut out of dark gloomy silence into dazzling bright mint-white sunburst)
- **Connects To:** `scene_06_three_jars` (mint-white radial sunburst sweeps smoothly outward into Scene 6's identical mint-white sunburst background)
- **Khmer Overlay (added in post):** `ឈប់សិន! សាកវិធីនេះ...`

```text
Use the attached image (scene_05_lightbulb_idea.jpg) as the starting frame. Duration: 5 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and snappy upbeat cartoon physics, no random scene cuts. Explosive contrast snap-cut out of the previous dark scene into this dazzling bright off-white and soft mint-green radial sunburst background. [0.0s–1.0s] Standing confidently with his right hand on his hip, sharp confident grin, and determined raised eyebrows, the young Cambodian man snaps the fingers of his raised left hand briskly as crisp black snap-accent lines burst outward; instantly the bright yellow cartoon lightbulb above his hand pops on with a radiant golden flash. [1.0s–3.2s] The mint-and-white radial sunburst stripes rotate smoothly and briskly behind him while warm golden light rays pulse around the glowing lightbulb, small golden four-point sparkles twinkle excitedly, and the character gives a confident energetic nod with a subtle snappy zoom-in. [3.2s–4.4s] He shifts his gaze cheerfully forward as his upbeat confidence builds. [4.4s–5.0s] Outgoing transition beat: the bright mint-white radial sunburst rays expand smoothly outward with a clean uplifting visual swoosh, matching the exact mint-white sunburst backdrop of the three-jars scene next. Sound design: crisp loud finger snap at 0.25s, bright resonant lightbulb ding at 0.5s, warm uplifting lo-fi acoustic chords entering cleanly at 0.6s, and a smooth upward swoosh transition at 3.9s–4.5s. Keep top and bottom zones open for captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 6 — `scene_06_three_jars` (`0:44 – 0:59` | Duration: **15s**)
- **Input Image:** `topic1_scenes/scene_06_three_jars.jpg`
- **Connects From:** `scene_05_lightbulb_idea` (seamless continuation of Scene 5's exact bright mint-white radial sunburst)
- **Connects To:** `scene_07_new_day25` (once the golden padlock clicks shut on the green jar, camera pushes softly toward the glowing green jar, dissolving directly into Scene 7's green jar with sprout)
- **Khmer Overlay (added in post):** `៥០% ចំណាយចាំបាច់` (`0.35s`) / `៣០% កម្សាន្ត` (`4.2s`) / `២០% សន្សំ` (`8.2s`)

```text
Use the attached image (scene_06_three_jars.jpg) as the starting frame. Duration: 15 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines and satisfying smooth motion-graphics physics, no random scene cuts. Seamless continuation of the bright off-white and soft mint-green radial sunburst from the previous scene, with gentle floating mint sparkles behind the smiling young Cambodian man standing behind the clean light-wood table with three clear glass jars lined up left to right. [0.0s–4.5s] Phase 1 (Left Blue Jar — Needs): holding his smartphone calmly with a warm smile, shiny golden coins launch in a smooth glowing arc from the smartphone into the left glass jar; inside, glowing blue coins stack higher with a gentle luminous shimmer while the tiny vector house and rice bowl icons bob cheerfully above the jar rim. [4.5s–8.5s] Phase 2 (Middle Orange Jar — Wants): a second neat glowing arc of golden coins streams rhythmically from the smartphone into the middle jar; warm orange coins fill upward smoothly as the tiny vector coffee cup and shopping bag icons bounce lightly above it, and the man nods with calm satisfaction. [8.5s–13.5s] Phase 3 (Right Green Jar — Savings): the third luminous stream of golden coins arcs cleanly into the right glass jar as bright emerald-green coins rise inside; at 12.6s the cute golden padlock on the right jar neck snaps firmly shut with a crisp sparkle burst and brief radiant glow around the green jar. [13.5s–15.0s] Outgoing transition beat: the character smiles proudly as the camera gently glides and pushes slightly inward toward the glowing locked green savings jar on the right, setting up a smooth visual match-dissolve into that same glowing green savings jar on the café table in the next scene. Sound design: steady feel-good warm lo-fi beat drops in smoothly at 0.0s, crisp rhythmic glass coin clink-clink-clink triplets synced to each jar phase (1.2s–2.2s, 4.2s–5.2s, 7.4s–8.4s, 10.0s–11.0s), followed by a crisp metallic padlock click at 12.6s and a bright rewarding bell chime at 12.85s. Keep upper and lower areas open for Khmer percentage badges. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 7 — `scene_07_new_day25` (`0:59 – 1:08` | Duration: **9s**)
- **Input Image:** `topic1_scenes/scene_07_new_day25.jpg`
- **Connects From:** `scene_06_three_jars` (match-dissolves directly from Scene 6's glowing green savings jar to the glowing green savings jar with fresh sprout)
- **Connects To:** `scene_08_outro` (warm mint-cream sunlight blooms softly and camera pulls gently back into Scene 8's airy mint-white outro)
- **Khmer Overlay (added in post):** `ថ្ងៃទី ២៥ នៅតែញញឹម 😌`

```text
Use the attached image (scene_07_new_day25.jpg) as the starting frame. Duration: 9 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Smooth warm visual match-transition from the previous scene's glowing green savings jar directly into this sunny pastel-cream and soft-mint café corner. Seated comfortably at the tidy wooden dining table with greenery outside the window and hanging potted plants gently swaying, the young Cambodian man holds his smartphone with relaxed shoulders and a genuinely peaceful smile. [0.0s–3.0s] Soft wisps of warm aromatic steam curl gently upward from the bowl of white rice and the savory Southeast Asian soup bowls while ice cubes bob subtly in his glass of iced drink; he checks his smartphone calmly and blinks with relieved contentment. [3.0s–6.5s] On the right side of the table, his glowing emerald-green glass savings jar filled with green banknotes and gold coins pulses with a gentle warm aura as the fresh green two-leaf plant sprout on top of its wooden lid grows slightly taller and sways happily with tiny golden sparkles rising softly. [6.5s–8.2s] Warm afternoon sunlight glows peacefully across the room as he smiles serenely, enjoying his comfortable end-of-month meal. [8.2s–9.0s] Outgoing transition beat: soft mint-cream light blooms gently from the background and the camera eases smoothly back, clearing the upper frame seamlessly into the clean airy mint-white outro layout next. Sound design: soft gentle page-flip whoosh at 0.1s, uplifting lo-fi acoustic groove reaching its warm melodic peak with shimmering bell chimes at 1.2s and 1.55s. Leave clean space at top and bottom for captions. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```

---

### Scene 8 — `scene_08_outro` (`1:08 – 1:15` | Duration: **7s**)
- **Input Image:** `topic1_scenes/scene_08_outro.jpg`
- **Connects From:** `scene_07_new_day25` (smooth airy mint-white bloom carrying the glowing sprouted green jar onto the outro shelf)
- **Connects To:** Clean Reels ending & seamless loop back to `scene_00_hook`
- **Khmer Overlay (added in post):** `ចេះបែងចែក ចុងខែមិនដាច់លុយ 💡` (`0.3s`) / `ដាច់លុយតាំងពីថ្ងៃទីប៉ុន្មាន? Comment 👇` (`2.2s`)
- **Optional Loop Tip:** For a perfectly loopable outro in Gemini Omni Flash, pass `scene_08_outro.jpg` as **both** `--first-frame` and `--last-frame` (supported up to `10s`).

```text
Use the attached image (scene_08_outro.jpg) as the starting frame. Duration: 7 seconds, 9:16 vertical. Single continuous 2D flat-vector cel-shaded animation in modern DEMIC motion-graphics style with crisp dark-brown outlines, no random scene cuts. Smooth airy transition out of the previous sunny café into this clean minimal pastel mint-white outro background, keeping the upper 55 percent of the frame completely open and uncluttered for bold takeaway captions. [0.0s–2.2s] Positioned in the lower-left third with left hand on hip and warm happy closed crescent eyes, the young Cambodian man raises his right hand in a friendly confident thumbs-up with a gentle upbeat bounce while his shoulders relax cheerfully. [2.2s–5.0s] Beside him on the minimal lower-right shelf, the three glowing colored glass jars (blue, orange, and green with the little two-leaf plant sprout on top) shimmer softly in sequence left-to-right as tiny golden sparkles drift delicately upward and the little green leaves sway gently. [5.0s–7.0s] Outgoing ending & loop beat: calm steady camera hold as the character gives one warm friendly blink and settled confident smile while the soft glow around the three jars pulses gently and settles cleanly into a peaceful ending frame ready to loop smoothly. Sound design: warm uplifting lo-fi acoustic music resolves gently with two soft playful pops at 0.35s and 2.4s, a clean sparkling chime tail at 4.0s, and a smooth 2-second musical fade-out from 5.0s to 7.0s. Strictly no spoken dialogue, no voiceover, no on-screen text, no letters, no numbers, no logos, no watermarks.
```
