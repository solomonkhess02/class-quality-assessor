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
