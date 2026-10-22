# Experiment log
Append-only. Each entry: hypothesis, method, result, decision. No real names or raw transcript text.

## E1 (21 Oct): What do the organisers actually judge?
- Method: read the organiser's feedback notes for all 10 labelled classes.
- Result: recurring criteria are intro and rapport before teaching, time for new joiners to introduce themselves, asking each child questions individually, giving homework, a proper conclusion, energetic and pleasant communication, branded 8x8 background, tutor face visible / camera on, lighting, and video quality.
- Decision: about half of these are only visible on screen, so a transcript-only system is ruled out. Rubric in docs/ARCHITECTURE.md is built from these points.

## E2 (21 Oct): Do simple transcript stats separate quality levels?
- Method: parsed the 12 Zoom VTTs; tutor = speaker with most talk time.

| label | class | min | tutor talk % | other speakers | student turns/min | tutor "?"/min | homework mentions | students named by tutor |
|---|---|---|---|---|---|---|---|---|
| average | B435 | 38 | 57 | 4 | 2.5 | 2.9 | 0 | 1 |
| average | B516 | 44 | 77 | 5 | 3.4 | 3.6 | 7 | 2 |
| demo | english | 44 | 60 | 2 | 7.9 | 5.0 | 0 | 0 |
| demo | malayalam | 52 | 82 | 1 | 4.0 | 1.8 | 0 | 0 |
| excellent | B472 | 56 | 47 | 6 | 9.3 | 3.2 | 3 | 5 |
| excellent | B491 | 53 | 63 | 8 | 10.1 | 3.4 | 0 | 1 |
| good | B435 | 44 | 46 | 7 | 4.3 | 0.1 | 0 | 0 |
| good | B552 | 46 | 57 | 4 | 9.1 | 3.8 | 3 | 1 |
| quality issue | B392 | 60 | 74 | 5 | 4.3 | 2.8 | 0 | 5 |
| quality issue | B441 | 84 | 56 | 10 | 3.7 | 2.5 | 0 | 10 |
| weak | B482 | 59 | 66 | 5 | 2.3 | 1.0 | 0 | 3 |
| weak | B558 | 46 | 70 | 3 | 5.9 | 3.0 | 0 | 2 |

- Result: student turns/min is the strongest single signal (excellent ~9-10, average/weak mostly 2-4). Misleading signals: "?" counts depend on transcript punctuation (good B435 = 0.1); keyword counts for homework do not match whether homework was assigned (average B516 = 7, excellent B491 = 0); name mentions are highest in a quality-issue class. The quality-issue classes look normal on text; their problems were visual. The Malayalam demo's Zoom transcript is English ASR applied to Malayalam speech and is unusable.
- Decision: stats are evidence for the judge, not the classifier. Homework and engagement need LLM judgement. Visual module is required. Transcript quality must be detected.

## E3 (22 Oct): Is whisper.cpp viable?
- Method: whisper.cpp b5454 CPU build, ggml-large-v3-turbo-q5_0, 2-minute English clip (03:00-05:00 of average B435), 6 threads, -l en; compared with the Zoom VTT by word alignment.
- Result: 141 s for 120 s of audio (1.18x real time), peak RAM ~0.9 GB, encoder 92% of runtime. 84% word agreement with the VTT (both are machine transcripts, so this is agreement, not accuracy). Whisper was right on chess terms where the VTT was wrong.
- Decision: quality is good but turbo is too slow for a 2-vCPU host. Next: benchmark base-q5_1 and small-q5_1 at 2 threads; transcribe only opening/closing/middle windows if needed. Turbo is not trained for translation, so Malayalam needs -l ml (LLM reads it) or a non-turbo model with -tr.

## E4 (22 Oct): Text stats module vs the E2 hand analysis (M3)
- Method: cqa/features/text_stats.py on all 12 VTTs (scripts/stats_table.py); same-speaker cues < 1.5 s apart are merged into one turn.
- Result: tutor talk share and tutor "?"/min match E2 within 1 point. Student turns/min is 10-35% lower than E2 because merged cues count once (excellent 7.3 / 9.4, good B552 8.3, average 1.6 / 3.2, weak 2.2 / 5.3, quality issue 3.9 / 2.9); the ordering is unchanged. good B435 is the only class where "?" counts are unreliable: "?" is 5% of its sentence-ending marks (21-43% elsewhere) and it has only ~31 tutor words/min (others 40-70). Longest tutor monologue is highest in weak B558 (447 s; others 30-154 s).
- Decision: the sparse_punctuation flag now fires on low punctuation density or a "?" share < 10% of sentence ends; the judge is told to ignore "?"/min when it is set. Low words/min is a candidate transcript-quality signal for Phase 2.

## E5 (22 Oct): Frame sampling, one decode pass vs per-timestamp seeks (M4)
- Method: scripts/bench_frames.py on the longest recording (quality issue B441, 89.4 min), 1 frame / 30 s at 640 px, dev laptop. Strategy A: one ffmpeg pass with a select filter (the fps filter rounds to the nearest slot and dropped the last frame in a synthetic test). Strategy B: one fast input seek (-ss before -i) per timestamp.
- Result: seeks 55.7 s, single pass 94.9 s; both 179 frames.
- Decision: per-timestamp seeks are the default (media.sample_frames). Re-check on the 2-vCPU Space.

## E6 (22 Oct): Do OpenCV frame metrics separate the classes? (M4)
- Method: scripts/visual_table.py on all 12 recordings; YuNet face detector on 640 px frames every 30 s. face% = frames with >= 1 face; dark% = mean grey < 60; frozen% = unchanged from the previous sample.

| class | label | face% | multi-face% | dark% | black% | frozen% | median blur |
|---|---|---|---|---|---|---|---|
| average B435 | Average | 41 | 1 | 28 | 0 | 26 | 852 |
| average B516 | Average | 86 | 1 | 58 | 1 | 3 | 864 |
| excellent B472 | Excellent | 87 | 0 | 17 | 1 | 5 | 1874 |
| excellent B491 | Excellent | 79 | 1 | 27 | 1 | 9 | 2016 |
| good B435 | Good | 40 | 0 | 48 | 0 | 32 | 783 |
| good B552 | Good | 65 | 2 | 6 | 0 | 11 | 1918 |
| quality issue B392 | Poor | 62 | 3 | 45 | 0 | 20 | 1314 |
| quality issue B441 | Poor | 4 | 1 | 42 | 0 | 31 | 1355 |
| weak B482 | Poor | 85 | 0 | 29 | 0 | 12 | 2197 |
| weak B558 | Poor | 71 | 0 | 17 | 0 | 15 | 604 |
| demo 1 | - | 79 | 6 | 26 | 0 | 20 | 1636 |
| demo 2 | - | 98 | 9 | 18 | 0 | 4 | 1437 |

- Result: quality issue B441 stands out (face in 4% of frames). Quality issue B392 does not stand out on face% (62); its dark% (45) is high but average B516 is darker (58), so dark% alone does not separate. Both B435 classes (average and good, same batch) show face ~40% with ~30% frozen frames, which looks like long screen shares of a static board rather than a missing tutor. Multi-face frames are <= 9% everywhere, so gallery view is rare; layout effects come from screen share. Black frames are ~0 everywhere.
- Decision: the "face missing > 40% -> at most Average" cap would wrongly cap good B435, so it becomes a warning until screen-share frames are tagged and excluded (contact-sheet check by eye, M4 follow-up). Dark threshold and frozen% are not trusted as penalties until then.

## E7 (22 Oct): Can the direct judge tell an excellent class from a weak one? (M6a)
- Method: direct design (LLM scores each criterion 1-4 with quotes), deepseek-flash, thinking disabled, temperature 0.2, VTT only, no visual metrics. excellent B472 vs weak B482.

| criterion | excellent B472 | weak B482 |
|---|---|---|
| opening_rapport | 3 | 2 |
| individual_engagement | 3 | 3 |
| communication | 3 | 3 |
| closure_homework | 3 | 2 |
| technical_quality | 2 | 2 |
| weighted mean (LLM criteria) | 2.88 (Good) | 2.47 (Average) |

- Result: the judge ranks the two classes correctly but compresses them into a 0.4 gap; both are one level off. Engagement and warmth, the criteria the organiser stresses most, get identical scores. All 33 quotes verified.
- Decision: this is a discrimination problem, not only calibration. Test an extract-then-score design before touching thresholds.

## E8 (22 Oct): whisper.cpp model and threads for a 2-vCPU host (M5)
- Method: scripts/bench_whisper.py, same 2-minute clip as E3 (03:00-05:00 of average B435), -l en, -mc 0, JSON output through cqa/transcript/whisper.py (window offsets applied). Agreement = word-sequence similarity with the Zoom VTT over the same window (both are machine transcripts). Dev laptop, 12 logical cores.

| model | threads | wall s | x real time | agreement % |
|---|---|---|---|---|
| large-v3-turbo-q5_0 | 6 | 125.7 | 1.05 | 83 |
| large-v3-turbo-q5_0 | 2 | 235.3 | 1.96 | 83 |
| small-q5_1 | 2 | 60.3 | 0.50 | 83 |
| base-q5_1 | 2 | 19.9 | 0.17 | 82 |

- Result: on this clip base-q5_1 matches turbo's agreement at about 1/12 of the 2-thread cost. A 50-minute class would take ~8.5 min with base and ~25 min with small on this laptop at 2 threads; the Space's vCPUs will likely be slower. Window offsets are correct (first segment at 180.0 s). E3 reproduced (turbo, 6 threads: 1.05x, 83%).
- Decision: base-q5_1 at 2 threads is the default (DECISIONS D11). One clip only and agreement is not accuracy; chess-term errors are not measured here.

## E9 (22 Oct): Extract-then-score vs direct scoring (M6b)
- Hypothesis: Likert scores bunch at 3; asking the LLM for observable facts (warm-up length, children greeted by name, children asked individually, praise count, homework, recap, goodbye, disruptions) and mapping them to 1-4 in code with rules taken from the anchor wording separates classes better.
- Method: both designs on excellent B472 and weak B482, VTT only, no visuals, same model and settings. Fact rules were written from the ARCHITECTURE anchors before looking at results and were not tuned afterwards.

| criterion | B472 direct | B472 facts | B482 direct | B482 facts |
|---|---|---|---|---|
| opening_rapport | 3 | 3 | 2 | 2 |
| individual_engagement | 3 | 3 | 3 | 4 |
| communication | 3 | 1 | 3 | 1 |
| closure_homework | 3 | 3 | 2 | 2 |
| technical_quality | 2 | 3 | 2 | 3 |
| weighted mean | 2.88 | 2.65 | 2.47 | 2.53 |

- Result: direct separates the two classes by 0.41, facts by 0.12 (and ranks the weak class higher on engagement). LLM-reported counts were implausible: 5 praises and 4 "negative tone" remarks in the excellent class, which drove communication to 1 for both. Yes/no facts did discriminate: homework assigned (yes vs no), children greeted by name before content (2 vs 0). All 60 quotes verified in both designs; generic observations 0-13%, so the single generic re-ask never fired.
- Organiser feedback (expected_criteria.json) for weak B482 praises individual questioning and faults rapport, new joiners' intro and camera; so direct's engagement 3 and opening 2 are consistent with it. The label error may come from aggregation rather than per-criterion judgement.
- Decision: keep the direct design (DECISIONS D10). Add the discriminating yes/no facts to the direct judge's output as evidence (M7). Measure criterion-level agreement and compare aggregation rules before changing the judge further.

## E10 (22 Oct): Criterion-level agreement and label aggregation on the labelled set (M7)
- Hypothesis (from E7/E9 and the organiser's per-criterion expectations): the judge is mostly right per criterion and the label error comes from aggregation.
- Method: scripts/eval_labelled.py, direct judge with key facts, two genuine runs per class (VTT + visual metrics; VTT only, where the judge never sees video numbers). Criterion agreement vs DATA_DIR/expected_criteria.json (35 non-null points; score >= 3 = good, <= 2 = weak). Two aggregation rules on the same criterion scores: (a) weighted mean with the ARCHITECTURE thresholds; (b) problem count = criteria scored <= 2 (opening and visual count fully), 0 Excellent, 1 Good, 2 Average, 3+ Poor, mapping set from the organiser's framing before looking at results. Full tables in eval/results.md.

| criterion | agreement, VTT + visual | agreement, VTT only |
|---|---|---|
| opening_rapport | 6/6 | 5/6 |
| individual_engagement | 6/7 | 7/7 |
| communication | 0/5 | 0/5 |
| closure_homework | 6/7 | 6/7 |
| visual_presence | 6/8 | not scored |
| technical_quality | 2/2 | 2/2 |
| all | 26/35 (74%) | 20/27 (74%) |

| rule | exact, VTT + visual | within one | exact, VTT only | within one |
|---|---|---|---|---|
| weighted mean | 1/10 | 9/10 | 3/10 | 8/10 |
| problem count | 4/10 | 8/10 | 5/10 | 7/10 |

- Result:
  1. Per criterion the judge mostly agrees with the organiser (74%), so the hypothesis holds.
  2. communication is 0/5: the judge gives "energy and warmth" 3 to 9-10 of 10 classes, and every disagreement is the organiser asking for more energetic/pleasant communication. Energy is mostly in the voice, which a transcript does not carry.
  3. technical_quality is 2 for nearly every class, including both excellent ones, so it adds a near-constant problem.
  4. The weighted mean never predicts Poor (0/4 in both modes): LLM scores of 1 are rare, so means compress into 2.15-3.45. The problem count finds 3 of 4 Poor classes in both modes, but is pessimistic (the constant tech problem makes Excellent nearly unreachable).
  5. Run-to-run noise: the two modes are separate judge calls with slightly different prompts, and some criteria moved by one point (e.g. average B435 engagement 2 vs 3).
  6. Confidence is 1.00 on 9 wrong labels (uninformative). Visual disagreements are quality issue B392 (missing 8x8 background, not measured in Phase 1) and weak B482 (camera "sometimes" off; face in 85% of frames).
  7. Bug found and fixed: with 0 scored criteria (unusable Malayalam transcript) the problem rule said Excellent; labels now need >= 3 scored criteria, otherwise a placeholder with a warning.
- Decision: ship the problem-count rule (DECISIONS D12) on reasoning plus results; thresholds are not fitted. Not done here, because it would be fitting to these 10 classes: dropping communication and technical_quality from the problem count. Phase 2 candidates: audio energy features for communication, a stricter definition for technical problems, confidence from label margin and run-to-run agreement, 8x8 background check.

## E11 (22 Oct): Do names still reach the judge on the VTT path? (P2.1)
- Hypothesis: replacing Zoom display names (exact + fuzzy on capitalised words) is not enough. Names said aloud that are nobody's display name, and lowercase variants, reach DeepSeek.
- Change: the VTT path now also runs the whisper-path [NAME] scrubber after display-name replacement (privacy.anonymise, used by the pipeline and the audit). The display-name fuzzy pass now also covers lowercase words (FUZZY_MIN 85, wider allowlist for lowercase).
- Method: scripts/leak_audit.py builds the exact judge input (VTT-only mode) for all 12 classes, with no API calls. Tokens from the fixed prompt template are ignored. (a) capitalised tokens not in the allowlist or pseudonyms; (b) tokens fuzzy-matching (>= 70) a token of the original display names, which are tokenised from the full name, independently of the pipeline; (c) [NAME] count and the share of tutor words replaced. Pass: (b) = 0 everywhere, (c) <= ~2%. Counts only; matched tokens go to .cache/leak_audit_hits.json for local review.

| class | (a) caps | (b) hits | (b) names hit | (c) [NAME] | (c) tutor share |
|---|---|---|---|---|---|
| average/B435 | 32 | 1 | 1 | 34 | 1.28% |
| average/B516 | 52 | 4 | 3 | 113 | 3.99% |
| demo/1 | 148 | 0 | 0 | 154 | 7.06% |
| demo/2 | 71 | 16 | 2 | 65 | 0.99% |
| excellent/B472 | 118 | 11 | 4 | 158 | 1.73% |
| excellent/B491 | 109 | 3 | 3 | 165 | 3.17% |
| good/B435 | 40 | 2 | 2 | 51 | 1.69% |
| good/B552 | 103 | 0 | 0 | 105 | 2.00% |
| quality issue/B392 | 92 | 12 | 4 | 128 | 2.05% |
| quality issue/B441 | 105 | 13 | 6 | 136 | 2.94% |
| weak/B482 | 43 | 0 | 0 | 50 | 2.25% |
| weak/B558 | 78 | 5 | 2 | 94 | 1.76% |

- Result:
  1. (b) fails in 9 of 12 classes. No hit is an exact match. Before the lowercase pass, excellent B472 had 2 hits at 85-99. The lowercase pass removed them (3 replacements in B472, 1 in demo/2), and every remaining hit scores 70-84. The fuzzy matcher puts ordinary words in this range when they are close to short names, so these need a manual look at the dump before they count as leaks. Every hit name token is one the pipeline already knows; the _base_name blind spot (device word first) did not occur.
  2. (c) is above 2% in 6 classes (B516 3.99%, B491 3.17%, B441 2.94%, B482 2.25%, B392 2.05%, Malayalam demo/1 7.06%, whose English ASR is noise). Aggregate breakdown of the tutor scrubs: repeated, vocatively used words (3+ uses, at least one followed by , ? !) are 3-49 scrubs per class and are very likely real children's names that are nobody's display name. Scrubs of words used only once (the upper bound on wrongly scrubbed ordinary words) are at most 1.39% of tutor words in every labelled class except B516 (2.75%) and demo/1 (5.66%).
  3. Rejected alternative: keep a capitalised word if it appears lowercase in the same class. It lowered (c) by only 0.1-0.8 points and, when one lowercase use was enough, it let display names back through (B516 (b) 4 -> 8).
- Decision: ship the second pass plus the lowercase fuzzy pass. The 70-84 hits wait for manual review of the dump. As defined, (c) counts correct scrubs of real names too; the measure of over-scrubbing is the one-off share, which fails only for B516 and the unusable Malayalam transcript. The judge input changed, so eval/results.md predates this change and has not been re-run.

## E11b (22 Oct): Choosing the scrubber settings against a human review of the leak hits (P2.1)
- Input: the user labelled the 37 distinct 70-84 matches from E11 (local only): 16 real names, mostly ASR-mangled spellings of known names, and 21 ordinary words. So E11's 70-84 band contained a real leak, not just noise.
- Change: fuzzy matching against display-name tokens is now case-insensitive. At >= 85 a match becomes the child's pseudonym (unchanged). In the new loose band (>= threshold, < 85) it becomes [NAME], not a pseudonym, so a mangled spelling is not credited to a specific child. Common English words (wordfreq zipf (en) >= cutoff) are skipped in the loose band and by the capitalised-word [NAME] scrubber; the chess and sentence allowlists stay. (c) is now the over-scrub share: common English words (zipf >= 3.5, fixed for all variants) that anonymisation replaced, as a share of tutor words, found by word-aligning the raw and anonymised turns.
- Method: scripts/scrub_sweep.py, grid set in advance: loose-band threshold 70 / 75 / 80 x common-word cutoff 3.5 / 4.0. Selection rule set in advance: catch all 16 labelled names with the fewest labelled words scrubbed; on a tie, the stricter privacy option. 11 classes (demo/1 skipped).

| variant | names caught (of 16) | labelled words scrubbed (of 21) | over-scrub max | over-scrub mean |
|---|---|---|---|---|
| E11 (no loose band, no skip) | 0 | 1 | 2.13% | 1.17% |
| >= 70, zipf >= 3.5 | 13 | 2 | 0.66% | 0.11% |
| **>= 70, zipf >= 4.0 (chosen)** | **15** | **4** | **1.11%** | **0.36%** |
| >= 75, zipf >= 3.5 | 7 | 2 | 0.66% | 0.11% |
| >= 75, zipf >= 4.0 | 8 | 4 | 1.06% | 0.36% |
| >= 80, zipf >= 3.5 | 5 | 1 | 0.66% | 0.11% |
| >= 80, zipf >= 4.0 | 5 | 1 | 1.06% | 0.36% |

- Result:
  1. No variant in the grid catches all 16. The name missed at zipf >= 4.0 is itself a common English word (zipf >= 4, 5 letters): the ASR turned a name into a real word, and any common-word skip at <= 4.0 keeps it. At 3.5, two more names of that kind are kept. Outside the grid, for information only and not used for selection: zipf >= 4.5 catches 16 with 5 words scrubbed and 1.51% max over-scrub; no skip catches 16 with all 21 words scrubbed.
  2. Chosen: >= 70 with zipf >= 4.0, the natural extension of the rule (most names caught, then fewest words).
  3. Bug found while auditing: the first version applied the common-word skip at every score, which let one common word at 89% similarity to a name through (B516, a new >= 85 hit). The skip now applies only to the loose band; >= 85 matches are replaced even when they are English words. The sweep was re-run after the fix.
  4. Over-scrub sources (aggregate): at zipf 3.5 almost all of it comes from the unchanged exact display-name path, where a display-name token is itself a common word (24 replacements in B472, 12 in B441). Some of those are probably correct scrubs of a child whose name is an English word, which the metric cannot tell apart. At 4.0, B516 adds 24 [NAME] scrubs of words in the zipf 3.5-4.0 band.
- Audit with the chosen settings (scripts/leak_audit.py; (b) columns count distinct tokens):

| class | (a) caps | (b) >= 85 | (b) 70-84 | labelled names left | unlabelled 70-84 | (c) over-scrub |
|---|---|---|---|---|---|---|
| average/B435 | 52 | 0 | 2 | 0 | 1 | 0.06% |
| average/B516 | 86 | 0 | 1 | 0 | 0 | 1.11% |
| demo/2 | 117 | 0 | 6 | 0 | 1 | 0.16% |
| excellent/B472 | 211 | 0 | 3 | 1 | 0 | 0.76% |
| excellent/B491 | 144 | 0 | 3 | 0 | 0 | 0.21% |
| good/B435 | 59 | 0 | 0 | 0 | 0 | 0.07% |
| good/B552 | 140 | 0 | 0 | 0 | 0 | 0.39% |
| quality issue/B392 | 139 | 0 | 1 | 0 | 0 | 0.08% |
| quality issue/B441 | 158 | 0 | 4 | 0 | 1 | 0.71% |
| weak/B482 | 66 | 0 | 1 | 0 | 1 | 0.26% |
| weak/B558 | 128 | 0 | 1 | 0 | 0 | 0.18% |

- Gate D16: FAIL. There are 0 hits at >= 85, but 1 labelled name is left (the common-word name in B472), and 4 new 70-84 matches are unlabelled. All 4 have zipf >= 4.33: they are words the skip now keeps instead of scrubbing them as mid-sentence capitals, so they still need labels. Over-scrub target <= 0.5%: missed in B516 (1.11%), B472 (0.76%) and B441 (0.71%). (a) rose because common capitalised words are no longer scrubbed.
- Caveat: these settings are tuned on our 12 classes and on one person's labels. Unseen classes will have different names, different ASR manglings and different common-word collisions, so the settings carry no guarantee beyond these classes. The common-word skip also applies to the whisper path, where any child whose name is a common English word (e.g. a nickname) is no longer scrubbed.
- Decision: ship >= 70 / zipf >= 4.0 as the best option inside the pre-set grid. Release stays blocked by D16 until the remaining name and the unlabelled matches are resolved.

## E11c (22 Oct): Display-name tokens that are common words (P2.1, timeboxed 20 min)
- Hypothesis: over-scrub in excellent B472 and quality issue B441 comes from display-name tokens that are English words, replaced in every lowercase use.
- Change: a display-name token with wordfreq zipf >= 4.5 (e.g. "Will") is replaced only when capitalised, both by the exact pattern and by the fuzzy pass. The audit applies the same rule (lowercase occurrences are not matched against such tokens). D16 now has the documented exception from E11b (a name the ASR turned into a dictionary word is accepted and counted in its own column), and the gate includes over-scrub, matching docs/EVALUATION.md.
- Result: only 6 classes have a display-name token at zipf >= 4.5, and it occurs in lowercase in tutor speech only in B472 (1x) and B441 (3x). Over-scrub: B472 0.76% -> 0.71%, B441 0.71% -> 0.62%, B516 unchanged at 1.11% (its over-scrub is [NAME] scrubs of words with zipf 3.5-4.0, E11b). Other classes stay at or below 0.39%. Gate D16: FAIL on over-scrub alone (3 classes above 0.5%); (b) >= 85 = 0, labelled names left = 0, 1 accepted word-like name, 4 unlabelled 70-84 matches awaiting review.
- Decision: keep the rule (it is correct: a lowercase verb is not the child). The over-scrub target is still missed after the timebox; numbers logged, moving on. Most of what is left comes from display-name tokens with zipf 3.5-4.5 and from [NAME] scrubs of zipf 3.5-4.0 words; some of it is probably correct scrubbing of names that are also words.

## E12 (22 Oct): Baseline v2, 3 runs, evaluation harness v2
- Method: scripts/eval_labelled.py --runs 3 --tag baseline, current pipeline unchanged (VTT + visual, deepseek-flash, direct design, problem-count rule, temperature 0.2, prompt hash 100cb76c421f, commit 179858d). Runs are separate judge calls (the run index is part of the cache key). Metrics as in docs/EVALUATION.md, mean [min-max] across runs. Full tables: eval/scorecard.md; row in eval/ablations.md.
- Result:

| metric | baseline v2 | Phase 1 (1 run) | target |
|---|---|---|---|
| Pairwise ordering (primary) | 66% [64-68%] | 71% | >= 80% |
| Exact / within one | 4.3 [4-5] / 7.7 [7-8] of 10 | 4 / 8 | - / >= 9 |
| MAE / bias | 0.80 [0.70-0.90] / -0.47 [-0.70 to -0.30] | 0.80 / -0.60 | <= 0.6 / +-0.3 |
| Weakness recall | 61% [61-61%] | 61% | >= 75% |
| Strength recall / balanced | 86% [82-88%] / 74% [72-75%] | 88% / 75% | - / >= 75% |
| Feedback recall (47 points) | 66% [62-68%] | - | >= 60% |
| Evidence validity / genericness | 100% / 12% [10-13%] | 100% / - | >= 95% / <= 20% |
| Run agreement | 8/10 | - | >= 8/10 |

  1. Noise band of the primary metric: 64-68% (4 points). Run-to-run variation is small: mean criterion score range 0.08, 8/10 labels identical in all runs. The unstable classes are excellent B472 (Good / Excellent / Average) and good B552 (Excellent / Excellent / Average).
  2. Phase 1's 71% pairwise lies above the band. Phase 1's 61% weakness recall lies inside it (all three runs give exactly 11/18). Between Phase 1 and now the judge input changed (the [NAME] second pass, the lowercase and loose-band fuzzy passes, and one line of the system prompt explaining [NAME]), so the 3-5 point drop may come from the privacy changes rather than noise. Not tested here. The whole drop comes from good B435: 3 problems in Phase 1, 4 in every run now, which puts it below average B516 (2) and level with Poor B441. With good B435 at 3, the three runs would score 69% / 72% / 74%.
  3. Feedback recall by source: transcript-observable 85% (28 points), voice-only 53% (6), video-observable 26% (12). Video points are the biggest gap, as expected without an 8x8 background check or camera-off stretches.
  4. Per criterion: communication weakness recall 0/5 in every run (as in E10). visual_presence catches 2 of 4 criticised points. opening_rapport strength recall varies most (33-67%).
  5. Confidence is 1.00 for every class, right or wrong, so it carries no information. The grader cited a line not in the report 0 times. The privacy gate fails on over-scrub (E11c).
- Follow-up (from the cached assessments, no API calls): the criterion that moved for good B435 is opening_rapport, 3 in Phase 1 and 2 in all three v2 runs (closure 1, visual 1, technical 2 unchanged). Phase 1's VTT-only run had already given it 2, so the class sits on the 2/3 boundary. Hypothesis that greetings now read "Hi [NAME]" and no longer count: not supported. The greetings use display-name pseudonyms in both versions, children greeted by name is 3 in both, and the personal-chat line Phase 1 cited contains no [NAME]; v2's judge describes the chat as "almost no personal conversation beyond 'how are you'". The second pass changed 8 opening-window turns (5 tutor and 6 student [NAME] tokens). No prompt change made for this.
- Decision: this is the reference for ablations; the keep rule uses the 4-point band. Before A0, check whether the privacy changes cost pairwise accuracy (one config change: run with the E11 scrubber, 3 runs) so the drop is not wrongly attributed to a later ablation. Grader spot-check for 3 classes: .cache/eval/baseline/feedback_spotcheck.md (local), awaiting the user's review.

## E13 (22 Oct): A0, evidence and observations before the score
- Change (JUDGE_EVIDENCE_FIRST=true): in the judge's output example each criterion lists observations, evidence and improvement before insufficient_evidence and score, so the model writes its reasons before the number. Same validation, same everything else. No extra [NAME] prompt line: the cache did not support the greeting hypothesis (E12 follow-up).
- Method: scripts/eval_labelled.py --runs 3 --tag a0_evidence_first, commit 190125d; reference = baseline (E12).
- Result (mean [min-max]; baseline in brackets): pairwise 69.9% [66.7-72.2] (66.2%); exact 5.0/10 [3-7] (4.3); within one 7.7 (7.7); MAE 0.73 (0.80); bias -0.07 [-0.20 to +0.10] (-0.47); weakness recall 46% [44-50] (61%); strength recall 90% (86%); balanced 68% (74%); feedback recall 63% (66%); run agreement 4/10 (8/10); mean criterion score range across runs 0.17 (0.08).
- Reading: scoring after the reasons makes the judge more lenient and less stable. It praises more (strength recall up) and catches fewer problems (weakness recall down 15 points), which removes most of the harsh bias. Good B435 improves from 4 to 3 problems and average B435 from 4 to 2; weak B558 drops to Good in two runs.
- Keep rule (operationalised before results: keep if pairwise > ref + 4 points, or within +-4 points and weakness recall up >= 2/18 or |bias| down >= 0.2, and no gate fails): pairwise +3.7 points is within the band, |bias| down 0.40, privacy gate pending review, not failed. **Kept.** The rule does not penalise the weakness-recall drop or the halved run agreement; both are flagged for A2, which bounds criterion scores with verified facts. The baseline noise band (4 points) probably understates this config's noise.

## E14 (22 Oct): A1, technical quality scored by code (D14) and communication out of the label (D15)
- D15 check (scripts/criterion_std.py on the A0 runs; per-class mean over runs, population std across classes): opening_rapport 0.43, individual_engagement 0.44, communication 0.10, closure_homework 0.86, visual_presence 1.04, technical_quality 0.55 as scored by the LLM and 0.43 with the code scorer. Only communication is below 0.3, so LABEL_EXCLUDE=communication: still scored and shown, left out of the problem count and weighted mean.
- Change (TECH_SCORER=code, LABEL_EXCLUDE=communication): technical_quality = 2 if >= 3 connectivity incidents per hour (turns matching the connectivity phrases, merged when < 60 s apart) or >= 20% frozen camera frames (frames neither screen share nor black), else 3; the judge's observations and verified quotes are kept as context. The judge input is unchanged from A0 (new measurements are kept out of the prompt), so A1's runs reused A0's judge calls and differ only in scoring.
- Result (mean [min-max]; A0 in brackets): pairwise 70% [68-72] (70%); exact 3.3/10 [3-4] (5.0); within one 8.3 [8-9] (7.7); MAE 0.83 (0.73); bias -0.30 [-0.40 to -0.20] (-0.07); weakness recall 46% (46%); balanced 68% (68%); feedback recall 62% (63%); run agreement 6/10 (4/10).
- Reading: exact accuracy fell as expected, but not for the expected reason. The code scorer finds a technical problem in 9 of 12 classes (both excellent ones included), so the near-constant technical problem is still there. Two measurement causes, both visible in per-class numbers: (1) the connectivity phrases count classroom routine ("on mute" / "unmute") and affirmatives ("can see you", "can hear"), giving >= 3 incidents per hour in 6 classes; (2) frozen camera share is >= 20% in 6 classes, mostly those where static board frames are not tagged as screen share (weak B558 has only 17% frames tagged as camera). The D14 thresholds were applied as fixed; the incident definition was not tightened after seeing which classes it moves.
- Decision: kept, as decided in D14/D15. Open for the user: define a technical incident as a problem phrase only (cannot hear/see, voice breaking, lagging, disconnected, frozen), excluding mute management and affirmatives, and count frozen frames only once screen-share tagging is validated (E6).

## E15 (22 Oct): A2, verified yes/no facts bound the opening, engagement and closure scores
- Change (FACTS_DRIVE=true): the judge also reports rapport_before_content and each_child_asked_individually (each true fact needs a verified quote, else it is reset to false), alongside homework_assigned, recap_present and new_joiners. Code clamps scores with rules written before the run: opening <= 2 without rapport or with a new joiner not invited, >= 3 with rapport and no such joiner; engagement >= 3 if every child is asked individually, else <= 2; closure >= 3 with homework, <= 3 without a recap, <= 2 with neither; cap wins on conflict; each change is added to the criterion's observations.
- Method: --runs 3 --tag a2_facts_drive, commit 7a5ab28, on top of A0 + A1; reference = A1.
- Result (mean [min-max]; A1 in brackets): pairwise 75.5% [67-85] (69.9%); exact 5.0/10 [3-6] (3.3); within one 8.0 (8.3); MAE 0.70 (0.83); bias -0.50 [-0.60 to -0.30] (-0.30); weakness recall 54% [50-56] (46%); strength recall 82% (90%); balanced 68% (68%); feedback recall 61% (62%); evidence validity 100%; run agreement 7/10 (6/10).
- Per criterion: opening weakness recall 78% (A1 56%), closure strength recall 100% (78%), engagement strength recall 71% (100%): the "every child asked individually" cap pulls engagement down in classes the organiser praised for it. All 4 Poor classes are now Poor in at least 2 of 3 runs; both Average and both Excellent classes are predicted one level low or more (average B435 Poor, excellent B491 Average).
- Keep rule: pairwise +5.6 points > 4-point band, privacy gate pending review (not failed). **Kept.** Costs: bias back to harsh (-0.50) and a wider run range (67-85%), so the improvement is less certain than the mean suggests.

## E16 (22 Oct): A3, severity-tiered two-sided label rule and boundary confidence
- Change (AGGREGATION=severity), as written in docs/EVALUATION.md before the run. Critical -> Poor: no rapport before content (the A2 fact), tutor face in < 60% of camera frames (VISUAL_MIN_FACE_PCT); "no 8x8 background" is not measured yet. Moderate: no homework, no closing recap, low energy (communication <= 2; never counts while D15 excludes communication), lighting (> 20% dark frames); 2+ -> Average, 1 -> Good. Excellent needs zero problems plus engagement 4 and opening >= 3, otherwise Good. Ordering value for the primary metric: 10 x critical + moderate (+0.5 for problem-free without positive evidence), so every Poor ranks below every non-Poor. Confidence: 0.5 + 0.25 x min(margin, 2), margin = fewest findings that must flip to change the label; capped at 0.25 for unusable input. score/100 = weighted mean of the counted criteria, 1-4 -> 0-100.
- Caveat: the rule was derived from the organiser's notes on these same 10 classes, so its accuracy here is optimistic and says little about unseen classes.
- Method: --runs 3 --tag a3_severity_rule, commit 5c41a28, on top of A0-A2 (judge input identical to A2; only the label rule differs). Reference = A2.
- Result (mean [min-max]; A2 in brackets): pairwise 84.3% [82-88] (75.5%); exact 5.0/10 [5-5] (5.0); within one 7.0 (8.0); MAE 0.80 (0.70); bias -0.80 (-0.50); weakness recall 54% (54%); feedback recall 61% (61%); run agreement 10/10 (7/10); confidence on correct 0.87 vs wrong 0.82 (with run agreement 0.93 vs 0.91).
- Confusion (summed over runs): all 12 Poor predictions for Poor classes are right; Average 3 right, 3 -> Poor; Good 6 -> Poor; Excellent 3 -> Average, 3 -> Good. The rule orders classes well but is harsh and never reaches Excellent.
- Still wrong: average B435 -> Poor (rapport fact false; face in 38% of camera frames); excellent B472 -> Good (no closing recap; no engagement 4); excellent B491 -> Average (no homework, no recap); good B435 -> Poor (face in 39% of camera frames, likely untagged board screen share, E6); good B552 -> Poor (rapport fact false although the organiser praised its opening).
- Keep rule: pairwise +8.8 points > band, privacy gate pending review (not failed). **Kept** (D18). Next: the two critical checks need better measurements (screen-share tagging for the face check; a missed rapport quote should not become a critical problem on its own), which is A4 territory.

## E17 (22 Oct): A3b, measurement fixes for the severity rule
- Changes (to measurements, not thresholds): (1) three-state facts: yes and no each need a verified quote showing them (value yes / no), otherwise unknown, which never bounds a score or fires a check; code guards move a "no" to unknown when the transcript contradicts it (rapport: any tutor greeting or addressed child before the quoted line; homework: homework / assignment / practice / "at home" in the closing window). (2) Face share below 60% of camera frames is a moderate problem plus a warning until the screen-share tag is validated (FACE_CHECK_CRITICAL=false, E6). (3) Lighting = median brightness of the largest face box on camera frames, not the whole frame; threshold fixed from the definition before the run: a face is underexposed when its luma is more than one stop below middle grey (18% grey = 118 in sRGB, 9% = 85), so a problem iff median face luma < 85; unknown with < 5 face frames. (4) Technical incidents count connectivity problem phrases only (can't hear, breaking, network / internet / bad connection, lag, frozen, disconnected, rejoin, screen not visible); no mute management, no "can you hear me", no affirmatives; D14 thresholds unchanged; the connectivity_mentions statistic sent to the judge is unchanged. (5) Label-free filter D19. (6) Post-hoc: positive evidence for Excellent = rapport_before_content yes AND each_child_asked_individually yes (was engagement 4 and opening >= 3), for consistency with A2.
- Method: measurement pass --tag a3b_rates --no-record (new judge calls, as the fact prompt changed) -> scripts/check_rates.py -> final --runs 3 --tag a3b_measurement_fixes (judge calls are cache hits from the measurement pass; commit cd5dbfc). Reference = A3.
- Check firing rates (classes of 12): no_rapport 0 (5 unknown), face_low 5, no_homework 6 (3 unknown), no_recap 9, low_energy 0, dark_face 5, technical problem 6 (informational; D14 phrases fixed, still 50%). None >= 80%, nothing excluded.
- Guards (36 assessments): rapport "no" -> unknown 14 times, i.e. every rapport "no" the judge gave had a tutor greeting or an addressed child before its quote; homework "no" -> unknown 6 times. Fact states after verification: rapport 19 yes / 0 no / 17 unknown; each child 26 / 2 / 8; homework 9 / 18 / 9; recap 7 / 26 / 3.
- Result (mean [min-max]; A3 in brackets): pairwise 75% [74-78] (84%); exact 3.0/10 (5.0); within one 9.3 [9-10] (7.0); MAE 0.77 (0.80); bias +0.03 [0.00 to +0.10] (-0.80); weakness recall 46% (54%); strength recall 88% (82%); feedback recall 61% (61%, above the 60% flag line); run agreement 9/10 (10/10); confidence correct 0.83 vs wrong 0.89 (fails "correct > wrong").
- Confusion: no class is predicted Poor. With rapport "no" never surviving the guard, the face check moderate and the 8x8 background unmeasured, the rule has no working critical check, so all 4 Poor classes become Average (the most problems any class reaches is 3 moderate). The Average classes are now right (6/6), good B552 is right (B552's Poor in A3 came only from an unverified rapport quote), and both Excellent classes are still a level low.
- Reading: the fixes remove all five too-harsh errors of A3 but also the false-positive criticals that A3's 84% partly rested on; the harsh errors are replaced by lenient ones on the Poor classes. Under the keep rule (pairwise -9 points) A3b would be rejected. It stays in the code because it is the correct measurement as decided; Poor detection now waits on two measured critical checks: the face check once the contact-sheet labels validate the screen-share tag (>= 85% agreement -> FACE_CHECK_CRITICAL=true; face_low fires for average B435, good B435, quality issue B392, quality issue B441 and weak B558), and the 8x8 background check (A4). Accuracy work is frozen here as agreed.

## E6b (22 Oct): E6 follow-up, screen-share tag against hand labels
- Method: the user labelled the 20 contact-sheet frames (10 tagged share, 10 tagged camera, random, non-black): share 9, both 6 (screen share plus a tutor camera thumbnail with a face), camera 4, other 1, camera_off 0. scripts/contact_sheet.py --score: agreement counts both and camera as "tutor visible" (the tag should say camera there); other excluded.

| tag \ hand label | share | both | camera | other |
|---|---|---|---|---|
| share | 7 | 2 | 1 | 0 |
| camera | 2 | 4 | 3 | 1 |

- Result: agreement 14/19 = 74%, below the 85% bar, so FACE_CHECK_CRITICAL stays false. Errors go both ways: 2 real screen shares tagged camera (good B552, quality issue B441) and 3 tutor-visible frames tagged share (excellent B472 x2, average B516). YuNet found a face in 6/6 "both" frames and 4/4 camera frames at 640 px, so the hypothesis that it misses small thumbnail faces is not supported; no full-resolution thumbnail check is proposed. The 3 sampled average B435 frames were all tagged correctly and good B435 was not sampled, so this sample does not explain the low face share of both B435 classes.
- Decision: keep the face check moderate. The weak part is the flat-block screen-share tag, not face detection. Next, if face matters again: a second sheet of ~20 camera-tagged frames without a face, stratified on the two B435 classes (about 10 minutes of labelling), to see whether those frames are shares, camera-off tiles or a tutor looking away.

## E18 (22 Oct): A3c, rapport measured from the warm-up before content
- Change (D20, thresholds fixed before results): the judge quotes the first line of lesson content (content_start). Code measures warm_up_min (first tutor greeting -> content_start) and warm_up_students (distinct students with a turn of >= 4 words in that window). rapport_before_content = "no" if < 2 students or < 2 min, "yes" if >= 3 students and >= 3 min, else unknown; no verified content_start -> unknown. Replaces A3b's greeting guard; the homework guard stays. Implementation choice not in the definition: no tutor greeting before content_start -> warm-up 0 min / 0 students (so "no").
- Bug found by the synthetic tests: quote verification also matched a quote against two adjacent turns, so when the turn before the quoted line was short, the pair scored 100 too and, as the earlier turn, won the tie; the verified timestamp then moved one turn early (it would have shortened every warm-up). Single-turn matches now win ties (the pair score carries a 1-point penalty), for all evidence.
- Method: --runs 3 --tag a3c_rapport_warmup, commit 80cb576 (the run was interrupted by an exhausted DeepSeek balance at 17/36 and resumed; stored assessments and cached judge calls were reused). Reference = A3b (kept by override, D20).

| class | true | warm_up_min / students (runs) | rapport fact | organiser opening expectation | notes/points mentioning rapport |
|---|---|---|---|---|---|
| excellent/B472 | Excellent | 8.1/2, 16/3, 16/3 | unknown, yes, yes | good | 2 of 5 |
| excellent/B491 | Excellent | 4.1/3, 14.1/7, 14.9/7 | yes x3 | - | 0 of 6 |
| good/B552 | Good | 5.5/2 x3 | unknown x3 | good | 1 of 5 |
| good/B435 | Good | 0/0 x3 | no x3 | good | 2 of 5 |
| average/B516 | Average | 0.9/1 x3 | no x3 | - | 0 of 6 (by keyword; the user reads its notes as early individual engagement) |
| average/B435 | Average | 1.6/0 x3 | no x3 | - | 0 of 8 |
| quality issue/B392 | Poor | 0.6/1 x3 | no x3 | - | 0 of 5 |
| quality issue/B441 | Poor | 4.7/1 x3 | no x3 | weak | 2 of 5 |
| weak/B482 | Poor | 2.7/1 x3 | no x3 | weak | 3 of 6 |
| weak/B558 | Poor | 1.9/1 x3 | no x3 | weak | 3 of 6 |

- Result (mean [min-max]; A3b in brackets): pairwise 77% [76-78] (75%); exact 5.0/10 (3.0); within one 8.7 (9.3); MAE 0.63 (0.77); bias -0.63 [-0.70 to -0.60] (+0.03); weakness recall 50% (46%); strength recall 90% (88%); balanced 70% (67%); feedback recall 66% (61%); run agreement 9/10; confidence correct 0.75 vs wrong 0.75 (with run agreement 0.88 vs 0.84).
- Reading: the warm-up separates the extremes (both Excellent classes 4-16 min with 2-7 children talking; all 4 Poor classes "no", 3 of them with < 2 min or 1 child), and Poor detection is back (12/12). But "no warm-up" does push both Average classes to Poor in every run (B435 1.6 min / 0 children, B516 0.9 min / 1 child), and good B435 to Poor. B516 fits the user's hypothesis that rapport can happen through early individual questioning rather than a separate chat; the rule is not changed. Good B435's 0/0 is a measurement artefact: its quoted content line is at 4:00 while the speaker identified as tutor has no turn before 6:26 (4 children spoke before content), so no tutor greeting is found and the no-greeting choice above makes it "no".
- Keep rule (reference A3b): pairwise +2.3 points is within the 4-point band; weakness recall +0.7 of 18 (< 2/18); |bias| 0.03 -> 0.63 (worse). **The rule picks A3b.** A3c stays in the code for now (rapport measured from the transcript rather than a guard), pending the user's decision; the two open measurement questions are the no-greeting case (should be unknown, by "unknown is never a problem") and tutor identification when the teaching account changes.

## E19 (22 Oct): pywhispercpp bindings vs whisper-cli for the hosted no-VTT path
- Hypothesis: Streamlit Community Cloud cannot run a compiled whisper-cli, but the pywhispercpp PyPI wheel (Linux x86_64, bundles whisper.cpp 1.8.5 with AVX2/FMA and its own OpenMP) gives the same transcript with ggml-base-q5_1 and our parameters (-l, -t 2, max-context 0).
- Method: E8 clip (average/B435, 03:00-05:00), Docker linux/x86_64 limited to 2 CPUs; whisper-cli v1.9.5 (= b5454) through cqa/transcript/whisper.py vs pywhispercpp 1.5.1 through cqa/transcript/whisper_py.py, same 16 kHz WAV; word agreement by sequence alignment; numbers only.
- Result: py 0.14x real time vs cli 0.21x; both detect en with -l auto; segments 21 vs 12 (the bindings split into shorter segments); py matches 95.2% of cli words (140/147). Timestamps agree in range (first 180.0 s, last end 300.2 vs 302.0 s).
- Decision: WHISPER_BACKEND=py on Community Cloud (no compile step); cli stays the default locally and in Docker, where the benchmarks of D11 were run. Segment-level differences matter only for evidence timestamps (+-90 s window), so no re-benchmark before the deadline.

## E20 (22 Oct): A3d, rapport unknown when no tutor greeting is found; final configuration
- Change (code only, judge calls are cache hits from A3c): when the verified first content line has no tutor greeting before it, the warm-up cannot be located, so rapport is unknown instead of 0 min / 0 students -> "no" (the A3c implementation choice flagged in E18). Decided by the user before the run.
- Method: --runs 3 --tag a3d_rapport_unknown. The first attempt crashed: the quality check now sends the Malayalam demo to whisper, the 3 parallel runs started 3 whisper processes at once, and whisper's Malayalam stderr broke cp1252 decoding on Windows (fixed: UTF-8 decoding for whisper and ffmpeg output); demo/1 was then transcribed once (2032 s for its windows, language ml) and the eval rerun, reusing finished assessments. Reference = A3b (kept by override, D20).
- Result (mean [min-max]; A3b / A3c in brackets): pairwise 88.4% [88-89] (75.0% / 77.3%); exact 5.0/10 (3.0 / 5.0); within one 9.7 [9-10] (9.3 / 8.7); MAE 0.53 (0.77 / 0.63); bias -0.53 (+0.03 / -0.63); weakness recall 50% (46% / 50%); balanced 73% (67% / 70%); feedback recall 66% (61% / 66%); run agreement 9/10; confidence correct 0.75 vs wrong 0.80 (still fails correct > wrong; left for the report step).
- Per class: good B435 moves from Poor to Average (warm-up unknown; face, homework and recap problems remain). Still wrong: both Average classes -> Poor (warm-up located and short: B435 1.6 min / 0 children, B516 0.9 min / 1 child; B516's notes suggest rapport through early individual questioning, a known limitation), excellent B472 -> Good (no closing recap), excellent B491 -> Good / Good / Average (no homework), good B435 -> Average.
- Keep rule: pairwise +13 points over A3b, beyond the band; no gate failed. **A3d is the final shipped configuration (D21).** Accuracy work is frozen.

## E21 (22 Oct): No-VTT path, windowed whisper, quality check, role tagging, Malayalam (P2.2, API-free parts + role tagging)
- Windowed transcription (scripts/bench_asr_windows.py; base-q5_1, 2 threads, language auto; first 12 min, last 10 min, three 3-min middle windows = 31 min of audio): excellent B472 838 s (0.45x real time, 64% word agreement with the VTT over the same windows), good B552 561 s (0.30x, 64%), average B516 808 s (0.43x, 63%; partly overlapped with 3 stray whisper processes), weak B482 275 s (0.15x, 73%). Time tracks output length (3886 vs 1186 words) more than audio length, and is 2-3x slower than the 2-minute E8 clip suggested. With frame sampling and the judge, 2 of 4 classes would exceed the 10-minute target; proposed cut: 8 + 6 + 2x2 min (18 min of audio, about 4-10 min per class), pending the user.
- Transcript quality check (cqa/transcript/quality.py, thresholds from definitions): unusable when > 10% of tutor words are not English words (wordfreq zipf 0) or < 10 words/min. On the 12 Zoom VTTs (scripts/quality_table.py) only the Malayalam demo is flagged (13.7% non-English vs at most 3.1% elsewhere); tutor words/min alone does not separate (weak B482 26 < Malayalam 36). An unusable VTT now falls back to whisper with a warning.
- Role tagging (LLM labels each whisper segment TUTOR / STUDENT; scrubbed text only), accuracy against the most-overlapping VTT speaker: B472 65% (always-tutor baseline 55%), B552 72% (baseline 64%). Too weak: whisper segments are ~2 s and do not follow speaker turns. WHISPER_ROLE_TAGGING stays false, so on the no-VTT path student turns and the warm-up are not measured.
- Malayalam (demo/1, 5-min clip from 10:00, -l ml, 2 threads): base-q5_1 315 s (1.05x real time), small-q5_1 889 s (2.96x); 0% of the output in Malayalam script for both. The auto-language window transcript of the same class (2032 s) is Latin, Tamil and Devanagari script, no Malayalam. Neither model is usable for Malayalam at this cost.
- Bugs found: whisper-cli can write invalid UTF-8 into its JSON (a multi-byte character split across tokens); its Malayalam stderr broke cp1252 decoding on Windows. Both fixed (replacement on decode, UTF-8 for subprocess output).
- Decision: no Malayalam model is recommended. For a non-English class the app should say the transcript is not usable and fall back to the video-only part of the assessment with low confidence, rather than send script-salad to the judge (proposal for the user). Window cut pending the user.

## E22 (22 Oct): Shorter whisper windows, language detected once, chess prompt (no-VTT path)
- Hypothesis: windows of 8 + 6 + 2 x 2 min (18 min of audio, was 12 + 10 + 3 x 3 = 31 min), the language detected once on the opening window and reused, and an initial prompt with chess vocabulary keep the no-VTT path within 6 min of whisper per class at 2 threads without lowering agreement with the VTT below the current 64% (E21, B472). Decision rule set by the user before the run: all three become defaults if both hold; otherwise the shorter windows ship anyway (speed matters more on the free host) and both results are logged.
- Method: scripts/bench_asr_windows.py, base-q5_1, 2 threads, whisper-cli v1.9.5; agreement = word-sequence similarity with the Zoom VTT over the same windows. B472 twice on the same windows: plain (language auto per window, no prompt) and detect-once + prompt. Then whisper-cli -dl (language detection only) on every B472 window with and without the prompt. Then the plain config on the four benchmark classes.
- Result, B472: plain 351 s (0.32x real time), 69%; detect-once + prompt 469 s (0.43x), 62%, language si. The opening window is detected as Sinhala (p 0.71) with and without the prompt (identical probabilities); middle and closing windows are English (p 0.76-1.00, one Telugu at 0.40). Locking in the opening's language therefore forced Sinhala onto the whole class. Per-window auto also transcribes B472's opening as Sinhala, which is the window rapport is judged from.
- Result, plain config, four classes: B472 357 s (0.33x, 69%), good B552 235 s (0.22x, 63%), average B516 198 s (0.18x, 63%), weak B482 133 s (0.12x, 74%). All within 6 min; agreement 63-74% vs 63-73% for the longer windows (E21). Run-to-run timing on B472: 351 vs 357 s.
- Decision: the detect-once + prompt package failed both conditions, so the shorter windows ship with language auto per window and no prompt (ASR_DETECT_LANGUAGE_ONCE / ASR_CHESS_PROMPT stay available, off). The prompt's own effect on text was not measured separately (confounded by the Sinhala lock-in). Cost of the shorter windows, measured in E23: events between windows are missed.

## E23 (22 Oct): Final evaluation on the frozen configuration, and no-VTT vs VTT path consistency
- Step 2 of the final plan (demo/1 transcribed by itself, A3d rerun, keep rule vs A3b, D21) was already complete in E20 and was not repeated.
- Final eval: scripts/eval_labelled.py --runs 3 --tag final at commit a798bb9 (A3d, D21; judge and scoring unchanged since E20, so every judge and grader call was a cache hit). Result identical to E20: pairwise 88% [88-89]; exact 5/10; within one 9.7 [9-10]; MAE 0.53; bias -0.53; weakness recall 50%; strength recall 96%; balanced criterion accuracy 73%; feedback recall 66%; evidence validity 100%; run agreement 9/10. Privacy gate: 0 hits >= 85, 0 labelled names left, 4 new 70-84 matches pending human review (D16), so the gate is not yet passed. The demo class (unlabelled, Malayalam) reused its E20 whisper transcript made with the old windows. eval/scorecard.md is the README's numbers.
- Path consistency (scripts/compare_paths.py; one no-VTT run each, video + windowed whisper with the E22 defaults, vs the 3 VTT runs): excellent B472 VTT Good x3 -> no-VTT Average (closure 3 -> 2; homework yes -> no; individual questioning yes -> unknown); weak B482 VTT Poor x3 -> no-VTT Average (opening 2 -> 1, engagement 3 -> 2, communication 3 -> 2; rapport no -> unknown; individual questioning yes -> unknown). Label agreement 0/2; the no-VTT path moves both classes towards Average.
- Reading: (1) without speakers the warm-up cannot be located, so rapport is unknown (A3d) and the critical "no rapport" check cannot fire; a Poor class whose only critical problem is rapport rises to Average. (2) The shorter windows miss events between windows: B472's homework was set at 47:02 and the closing window starts at 50:00; its first content line (8:24) falls just after the 8-min opening window. A 10-min closing window would have caught the homework at about +1 min of whisper time per class (estimate from E22 rates, not measured).
- Decision: accuracy work stays frozen (D21). The no-VTT path is reported as less reliable, with these two causes, in the README limitations; the app already warns that speakers are not identified. Window lengths stay as decided by the E22 rule unless the user chooses otherwise.

## E24 (22 Oct): Names that are also common words, after the review of the last 4 matches (D16)
- Context: the user labelled the 4 unlabelled 70-84 matches from E23: 3 names, 1 word. The 3 names are names in their own right and also common words, so the scrubber's zipf skip kept them, and the audit's word-like-name exception accepted them; the gate read PASS with 4 accepted word-like names (average B435, excellent B472, quality issue B441, weak B482).
- Change (specified by the user before the run): in the 70-84 band, a token capitalised mid-sentence (not at the start of a turn or after . ? !) that matches a display-name token becomes [NAME] whatever its zipf; chess allowlist and TUTOR / S# / [NAME] unchanged. The audit accepts a labelled word-like name only if all its remaining occurrences are lowercase (D16 narrowed).
- Result (scripts/leak_audit.py, counts only): hits >= 85: 0; unlabelled 70-84: 0; labelled names left: 2, accepted lowercase-only: 0. **Gate D16: FAIL.** The 2 names left are one occurrence each, both capitalised at a sentence boundary, which the narrow rule deliberately keeps: excellent B472, at the start of a student turn; weak B482, in a tutor turn after . ? ! and followed by a comma (vocative). Average B435 and quality issue B441 are now clean. Over-scrub (monitored, D17): average B435 0.06% -> 0.23%, weak B482 0.26% -> 0.60%, quality issue B441 0.62% -> 0.65%; the newly replaced words are the reviewed names, which count as over-scrub because they are also common words. 4 classes over 0.5% (was 3).
- Decision: no lowercase-only exception applies, so none is documented; the gate fails and blocks release (D16) until the user decides. Candidate (not applied): also scrub a sentence-start token in this band when it is followed by , ? ! (the vocative rule scrub_names already uses), which would cover B482 but not B472's turn-start case.
- Follow-up, decided by the user: (1) the vocative rule is applied in the 70-84 band (sentence-start match followed by , ? ! -> [NAME]; without them it is kept), consistent with the whisper path. (2) B472's remaining token matches the tutor's display-name token, not a child's: an ASR-mangled form of the tutor's name rendered as a different name, one occurrence. It is a D16 documented exception, with no new rule; the audit accepts it only after checking that the matched token is the tutor's (it is).
- Result after the follow-up: hits >= 85: 0; labelled names left: 0; unlabelled 70-84: 0; documented exceptions: 1 (excellent B472). **Gate D16: PASS.** Weak B482 is clean. Over-scrub: weak B482 0.60% -> 0.66% (the reviewed name, also a common word); 4 classes over 0.5% (monitored, D17).

## E25 (22 Oct): No-VTT path, explicit language, 10-min closing window, judged rapport without speakers (supersedes E23's path-consistency result)
- Changes, decided by the user from E22/E23 before this run (E23 stays as written; this entry replaces its path-consistency numbers):
  (1) no language auto-detection: WHISPER_LANGUAGE=en by default and a class-language selector in the UI (English, Malayalam) passed to whisper -l. Reason: auto-detection labelled B472's English opening as Sinhala (E22); detect-once and the chess prompt stay off. The whisper cache file name now includes the language and window lengths.
  (2) ASR_CLOSING_MIN=10, matching the judge's 10-min closing window (closing_window_min) for consistency; opening stays 8 (windows 8 + 10 + 2 x 2 = 22 min of audio).
  (3) when speakers are unknown (no tutor-role turns, i.e. whisper without role tagging), the warm-up cannot be measured, so the judge is asked for rapport_before_content with the A3c thresholds (yes: about 3+ min of personal chat with several children answering before content; no: content within about 2 min of the first greeting with no chat). The rule is added to the user message only, so the VTT path's messages are unchanged (B472 and B482 VTT judge calls verified as cache hits, labels unchanged). Guards, fixed before results: a verified quote of the same value, in the opening, no later than the verified content start; a "no" also needs a verified content start inside the transcribed opening window; otherwise unknown. KeyFacts.rapport_method records warm_up / judge / none; the report row and a warning say when the judge method was used.
- Method: scripts/bench_asr_windows.py (whisper-cli v1.9.5, base-q5_1, 2 threads, -l en) on B472 and B482, then scripts/compare_paths.py --tag final (one no-VTT run each vs the 3 VTT runs).
- Result, whisper: B472 282 s for 22 min of audio (0.21x real time, 72% agreement with the VTT; E22 with auto: 357 s for 18 min, 69%), B482 172 s (0.13x, 74%; was 133 s, 74%). Forcing English removed the Sinhala decoding of B472's opening, which was slower than English.
- Result, path consistency: same label on 2 of 2 (was 0 of 2). B472: VTT Good x3, no-VTT Good; homework now yes (set at 47:02, inside the 10-min closing window); rapport yes via the judge (VTT: warm-up measured, between thresholds, unknown). B482: VTT Poor x3, no-VTT Poor; rapport no via the judge, as on the VTT path. Remaining difference on both: individual engagement 3 -> 2 (each_child_asked_individually yes -> unknown: individual questioning cannot be attributed without speakers); B472 children greeted by name 2 vs 3.
- Caveat: 2 classes, and the three changes were chosen after looking at these same two classes in E23, so 2/2 is optimistic; the thresholds and guards themselves were fixed before this run. Accuracy work stays frozen for the VTT path (D21).

## E26 (23 Oct): Tutor lines credited to children in verified quotes (live test, good B552)
- Report (live VTT test of B552): key-fact quotes of the tutor ("This will be your homework...", "next one, I need [NAME] to answer") were shown with student labels (S2, S4). Hypothesis given with the report: the tutor appears under a second Zoom display name and the most-talk-time rule makes it a student.
- Check 1, speakers in all 12 VTTs (counts only, scratch script): B552 has no tutor-like second speaker (#2: 21% of talk, 1% long turns, 5 instruction turns), and in the parsed VTT both quoted lines belong to the tutor (#1); no VTT has a cue without a speaker prefix. The hypothesis does not explain B552. A rule ">= 15% of talk with mostly long turns" would fire on average B435 #2 (35% talk, 83% long turns), excellent B472 #3 (21%, 70%) and quality issue B441 #2 (28%, 75%); in B435 and B472 the tutor keeps talking throughout that speaker's span (45 and 122 tutor turns), in B441 the tutor is nearly silent during it (31 s in 12 min) while the speaker has turns up to 258 s. Counts cannot tell a second tutor account from a co-teacher, a parent or played audio, so that part of the rule was not applied. No display name is >= 85 similar to its tutor's (highest 65).
- Check 2, the stored B552 assessments: 17 of 62 verified quotes carried a student label. For every misattributed one, the matched turn was a student turn of 1-4 characters (e.g. "a", "ok"): fuzz.partial_ratio aligns the shorter string inside the longer, so such a turn scores 100 against any quote containing those letters, and, coming before the real tutor line (also 100), it won the tie. The bug is in quote verification (cqa/judge/judge.py), not in role assignment.
- Fix (c86ec9c): a turn shorter than the quote is compared as a whole (fuzz.ratio), not as a window; a two-turn match is credited to the turn holding more of the quote. Plus the name part of the requested rule: a speaker whose display name matches the tutor's at >= 85 (device suffix stripped) is the tutor's second account (fires on none of the 12 classes). Tests: synthetic one-letter student turn before a tutor line; a quote spanning two turns; a two-account tutor.
- Re-run: scripts/eval_labelled.py --runs 3 --tag final_v2 (eval/scorecard.md; the previous card is kept as eval/scorecard_final.md). Confounded: the judge prompt also changed since `final` (E25 prompt changes; prompt hash 997d6ef78f15 -> b2eae1c92849), so judge calls were new. Result vs final: pairwise 88% -> 83%; exact 5 -> 4/10; within one 9.7 -> 9.7; MAE 0.53 -> 0.63; bias -0.53 -> -0.43; weakness recall 50% -> 48%; strength recall 96% -> 96%; balanced 73% -> 72%; feedback recall 66% -> 65% (T 86% -> 87%, V 21% -> 24%, A 56% -> 39%); evidence validity 100% -> 99%; run agreement 9/10 -> 9/10. Only one label changed: quality issue B441 Poor x3 -> Average x3. Its verified first content line now sits on the tutor's turn instead of the short turn before it, so the warm-up is 6.1 min with 3 children (was 4.7 min / 1) and "no rapport" no longer fires. Excellent B472 and B491 warm-ups also moved (8.1 -> 8.7 min, 4.1/3 -> 5.1/4) without a label change.
- Decision: keep the fix; it corrects a measurement, and as in D20 a bug is not kept because it scores better. Accuracy rules stay frozen (D21). The README shows final_v2.

## E27 (23 Oct): No-VTT path privacy and severity, after the live mp4-only test of good B552
- Report (live, mp4 only): a name-like token ("X, is it?" at 00:16:16) appeared in the strengths text, so it had reached DeepSeek; and B552 was Poor on the no-VTT path vs Good on the VTT path.
- Audit extended (scripts/leak_audit.py, --path vtt|whisper|both, default both): the whisper path is audited for every labelled class with a cached whisper transcript for the current window and language settings, by building the exact no-VTT judge input and matching it against that class's VTT display names (same bands; whisper-path review labels go under "whisper:<class>" and fall back to the class's VTT labels). D16 now applies to both paths.
- Scrubber, no-display-name path only (whisper, or a transcript without speaker names; the VTT path is unchanged): (a) a capitalised non-allowlisted word followed by , ? or ! is scrubbed even if it is a common word (vocative); (b) a capitalised word that is not common English (zipf < 4.0) is scrubbed even at a sentence start. Chess allowlist and TUTOR / S# / [NAME] unchanged. Synthetic tests.
- Result, audit after (a)+(b) (counts only): VTT path unchanged, PASS with 1 documented exception. Whisper path, 4 inputs (B472, B482, B552, demo/1 not gated): B482 and B552 clean; excellent B472 has 1 token at >= 85 to a child's display-name token: lowercase, mid-sentence, not an English word (zipf 0), so neither (a) nor (b) applies. **Gate D16: FAIL** (whisper path). The B552 phrase from the live report is not in the local B552 whisper transcript (the live app transcribes with the pywhispercpp bindings, this machine with whisper-cli; E19), so that occurrence could not be re-checked here; its shape is covered by the synthetic test.
- Candidate (c), measured, not applied (it is beyond the rules set for this step): on the whisper path, also scrub lowercase words of >= 3 letters that are not English at all (zipf 0) and not allowlisted. It would replace 4 of 1712 lowercase words in B472 (0.23%), 1 of 1192 in demo/1 and 0 in B482, and clears the B472 hit (1 -> 0).
- Severity (D22): the judge-fallback "no rapport" fact (rapport_method = judge, no speakers) counts as moderate, never critical. Test with synthetic facts.
- Path consistency (scripts/compare_paths.py --tag final_v2, one no-VTT run each vs the 3 final_v2 VTT runs, D22 and the new scrubber in place): excellent B472 Good x3 -> no-VTT Good (match); weak B482 Poor x3 -> no-VTT Average (was Poor in E25: its only critical problem was the judged "no rapport", now moderate, plus no homework and no recap = 3 moderate); good B552 Good x3 -> no-VTT Average (judged "no rapport", now moderate, plus lighting = 2 moderate; without D22 it would be Poor, as in the live test). Agreement 1/3 (E25: 2/2 on B472 and B482 before D22).
- Reading: D22 trades a harsh error (B552 Poor) for a lenient one (B482 Average); on this path a Poor label now needs a measured critical problem, which without speakers only the face check could provide once validated. The judged rapport fact disagrees with the VTT path on B552 (no vs unknown) and agrees on B482 (no).
- Decision: D22 stays (evidence quality, decided before the run). The whisper-path gate fails until the user decides on (c) or a documented exception.
- Follow-up, decided by the user: (c) applied on the no-transcript path only and only for an English class (transcript language en; Latin-script lowercase words of >= 3 letters, zipf 0, not allowlisted -> [NAME]). A Malayalam (or any non-English) class never passes through this scrub, and the pattern cannot match non-Latin script; tested with synthetic Malayalam text. Re-run (both paths, counts only): hits >= 85: 0; labelled names left: 0; unlabelled 70-84: 0; documented exceptions: 1 (VTT, E24). **Gate D16: PASS on both paths.** Residual risk: a name that whisper writes in lowercase and that has some frequency in the English word lists (zipf > 0) still passes on this path. D22 accepted; final_v2 stays the reported result.

## E28 (23 Oct): One-word-sentence names on the no-transcript path (live mp4-only test, good B552)
- Report: B552 mp4-only came out Average (D22 working), but one name still reached the judge: the tutor says it as a one-word sentence ending in a full stop ("<Name>. [NAME], your mic is off.", 00:26:51). Rule (a) covers only , ? ! and the token is a common word (zipf >= 4.0), so neither (a), (b) nor (c) applied.
- Fix (decided by the user; no-transcript path, English class only, after (c)): a capitalised word that forms a whole one-word sentence (at a turn or sentence start, followed by . or the end of the turn) becomes [NAME], unless it is an interjection (okay, ok, yes, no, yeah, good, great, nice, right, correct, wrong, perfect, excellent, hello, hi, bye, sorry, thanks, wow, done, next, wait, listen, look, see, now, so, well), a chess term or a pseudonym. Synthetic tests: "Name." and a turn-final "Name" scrubbed; "Okay.", "Checkmate.", "Right.", "Wait." kept; a Malayalam class untouched.
- Result, leak audit on both paths (counts only): **Gate D16: PASS** (hits >= 85: 0, labelled names left: 0, unlabelled 70-84: 0, documented exceptions: 1). Capitalised tokens left in the whisper-path judge inputs: B472 12 -> 11, B552 17 -> 16, demo/1 37 -> 34, B482 unchanged. As in E27, the live phrase itself could not be re-checked locally (different whisper build); its shape is covered by the tests.
- Also: the lighting check is phrased in plain words in Key reasons ("Lighting: the tutor's face is slightly underexposed (brightness 82, threshold 85)").
